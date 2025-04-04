# -*- coding: utf-8 -*-
import torch


class CRF(torch.nn.Module):
    """Linear-chain conditional random field.

    Given the source sequence $x = \{x_1, x_2, \dots, x_T \}$ and the target
    sequence $y = \{y_1, y_2, \dots, y_T \}$, the linear-chain CRF models the
    conditional probability as:
    $$
    P(y|x) = \frac{\exp \left( \sum_{t=1}^T U(x_t, y_t) + \sum_{t=0}^T T(y_t, y_{t+1}) \right)}{Z(x)}
    $$
    where $U(x_t, y_t)$ is emissions scores, $T(y_t, y_{t+1})$ is transition
    scores, and $Z(x_t)$ is partition function (a normalization factor).

    Hence, the negative log-likelihood loss (NLL-Loss) is:
    $$
    -\log \left( P(y|x) \right) = \log \left( Z(x) \right) - \left( \sum_{t=1}^T U(x_t, y_t) + \sum_{t=0}^T T(y_t, y_{t+1}) \right)
    $$

    Attributes
    ----------
    sos_transitions: nn.Parameter
        ``sos_transitions[j]`` is the score of transitioning from ``<sos>`` to ``j``.
    transitions: nn.Parameter
        ``transitions[i, j]``  is the score of transitioning from ``i`` to ``j``.
    eos_transitions: nn.Parameter
        ``eos_transitions[i]`` is the score of transitioning from ``i`` to ``<eos>``.

    Args
    ----
    emissions: torch.Tensor
        (step, batch, tag_dim)
    tag_ids: torch.LongTensor
        (step, batch)
    mask: torch.BoolTensor
        (step, batch)

    References
    ----------
    https://github.com/kmkurn/pytorch-crf
    """

    def __init__(self, tag_dim: int, pad_idx: int = None, batch_first: bool = True):
        super().__init__()

        self.sos_transitions = torch.nn.Parameter(torch.empty(tag_dim))
        self.transitions = torch.nn.Parameter(torch.empty(tag_dim, tag_dim))
        self.eos_transitions = torch.nn.Parameter(torch.empty(tag_dim))

        torch.nn.init.uniform_(self.sos_transitions.data, -0.1, 0.1)
        torch.nn.init.uniform_(self.transitions.data, -0.1, 0.1)
        torch.nn.init.uniform_(self.eos_transitions.data, -0.1, 0.1)

        if pad_idx is not None:
            self.sos_transitions.data[pad_idx] = -1e4
            self.transitions.data[pad_idx, :] = -1e4
            self.transitions.data[:, pad_idx] = -1e4
            self.eos_transitions.data[pad_idx] = -1e4

        self.tag_dim = tag_dim
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def extra_repr(self):
        return f"tag_dim={self.tag_dim}, pad_idx={self.pad_idx}, batch_first={self.batch_first}"

    def forward(
        self, emissions: torch.Tensor, tag_ids: torch.LongTensor, mask: torch.BoolTensor
    ):
        """
        Compute the negative log-likelihood, i.e., the loss.
        """
        if self.batch_first:
            emissions = emissions.permute(1, 0, 2)
            tag_ids = tag_ids.permute(1, 0)
            mask = mask.permute(1, 0)

        log_scores = self._compute_log_scores(emissions, tag_ids, mask)
        log_partitions = self._compute_log_partitions(emissions, mask)
        return log_partitions - log_scores

    def decode(self, emissions: torch.Tensor, mask: torch.BoolTensor):
        if self.batch_first:
            emissions = emissions.permute(1, 0, 2)
            mask = mask.permute(1, 0)

        return self._viterbi_decode(emissions, mask)

    def _compute_log_scores(
        self, emissions: torch.Tensor, tag_ids: torch.LongTensor, mask: torch.BoolTensor
    ):
        """
        Compute the numerator of the conditional probability in log space.
        """
        step, batch_size, tag_dim = emissions.size()
        batch_arange = torch.arange(batch_size, device=emissions.device)

        # Note: The first elements are assumed to be NOT masked.
        # log_scores: (batch, )
        log_scores = (
            self.sos_transitions[tag_ids[0]] + emissions[0, batch_arange, tag_ids[0]]
        )

        for t in range(1, step):
            # Transition -> Emission
            next_log_scores = (
                log_scores
                + self.transitions[tag_ids[t - 1], tag_ids[t]]
                + emissions[t, batch_arange, tag_ids[t]]
            )

            # Preserve the values where masked.
            log_scores = torch.where(mask[t], log_scores, next_log_scores)

        log_scores = (
            log_scores
            + self.eos_transitions[tag_ids[step - 1 - mask.sum(dim=0), batch_arange]]
        )
        return log_scores

    def _compute_log_partitions(self, emissions: torch.Tensor, mask: torch.BoolTensor):
        """
        Compute the denominator of the conditional probability in log space.
        """
        step, batch_size, tag_dim = emissions.size()

        # Note: The first elements are assumed to be NOT masked.
        # log_partitions: (batch, tag_dim)
        log_partitions = self.sos_transitions.expand(batch_size, -1) + emissions[0]

        for t in range(1, step):
            # Transition -> Emission
            # log_partitions: (batch, tag_dim) -> (batch, tag_dim, 1)
            # emissions[t]: (batch, tag_dim) -> (batch, 1, tag_dim)
            # next_log_partitions: (batch, tag_dim, tag_dim) -> (batch, tag_dim)
            next_log_partitions = (
                log_partitions.unsqueeze(2)
                + self.transitions
                + emissions[t].unsqueeze(1)
            ).logsumexp(dim=1)

            # Preserve the values where masked.
            log_partitions = torch.where(
                mask[t].unsqueeze(-1), log_partitions, next_log_partitions
            )

        # log_partitions: (batch, tag_dim) -> (batch, )
        log_partitions = (log_partitions + self.eos_transitions).logsumexp(dim=1)
        return log_partitions

    def _viterbi_decode(self, emissions: torch.Tensor, mask: torch.BoolTensor):
        """
        Decode the best paths.
        """
        step, batch_size, tag_dim = emissions.size()

        # Note: The first elements are assumed to be NOT masked.
        # log_best_scores: (batch, tag_dim)
        log_best_scores = self.sos_transitions.expand(batch_size, -1) + emissions[0]
        # history: list of ``indices`` of shape (batch, tag_dim)
        # In the ``k``-th example of a batch, at timestep ``t``, for each ``j``,
        # the best transition is from ``history[t][k, j]`` to ``j``.
        history = []

        for t in range(1, step):
            # Transition -> Emission
            # log_best_scores: (batch, tag_dim) -> (batch, tag_dim, 1)
            # emissions[t]: (batch, tag_dim) -> (batch, 1, tag_dim)
            # next_log_best_scores: (batch, tag_dim, tag_dim) -> (batch, tag_dim)
            # indices: (batch, tag_dim)
            next_log_best_scores, indices = (
                log_best_scores.unsqueeze(2)
                + self.transitions
                + emissions[t].unsqueeze(1)
            ).max(dim=1)
            history.append(indices)

            # Preserve the values where masked.
            log_best_scores = torch.where(
                mask[t].unsqueeze(-1), log_best_scores, next_log_best_scores
            )

        # log_best_scores: (batch, tag_dim) -> (batch, )
        # last_indices: (batch)
        log_best_scores, last_indices = (log_best_scores + self.eos_transitions).max(
            dim=1
        )

        # Retrieve the best paths
        best_paths = []
        for k in range(batch_size):
            # the best tag transitioning from to ``<eos>``
            best_path = [last_indices[k].item()]

            # retrieve the best path backward
            for indices in history[: step - 1 - mask[:, k].sum()][::-1]:
                best_path.append(indices[k, best_path[-1]].item())

            # reverse the order of best path
            best_paths.append(best_path[::-1])
        return best_paths
