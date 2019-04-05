"""
Structured perceptron implementation.
"""
from typing import List, Tuple, Dict, Optional

import torch

from allennlp.common.checks import ConfigurationError


class StructuredPerceptron(torch.nn.Module):
    def __init__(self) -> None:
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        super().__init__()

    def _score(self,
               unary_potentials: torch.Tensor,
               binary_potentials: torch.Tensor,
               tags: torch.Tensor,
               mask: torch.LongTensor) -> torch.Tensor:
        """
        Computes the score. Returns a Tensor of size [batch_size] where each
        element is the score for the sequence at that index.
        """

        batch_size, _, _ = unary_potentials.size()
        result = torch.zeros([batch_size])
        
        index = 0
        for pred, predMask, tag in zip(unary_potentials, mask, tags):
            score = 0
            seqLen = torch.sum(predMask)
            for i in range(seqLen):
                if i == 0:
                    score += pred[i][tag[i]]
                else:
                    score += pred[i][tag[i]] + binary_potentials[tag[i-1]][tag[i]]
            result[index] = score
            index += 1
        return result.to(self._device)

    def forward(self,
                unary_potentials: torch.Tensor,
                binary_potentials: torch.Tensor,
                tags: torch.Tensor,
                predicted_tags: List,
                mask: torch.ByteTensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        predicted_tags = torch.nn.utils.rnn.pad_sequence(
                list(torch.tensor(tags) for tags in predicted_tags),
                batch_first=True).to(self._device)
        gold_scores = self._score(unary_potentials, binary_potentials, tags, mask)
        pred_scores = self._score(unary_potentials, binary_potentials, predicted_tags, mask)

        # Hinge loss.
        losses = pred_scores - gold_scores
        losses = torch.max(losses, torch.FloatTensor([0]).expand_as(losses).to(self._device))
        loss = torch.sum(losses)

        return loss

    def get_tags(self,
                 unary_potentials: torch.Tensor,
                 binary_potentials: torch.Tensor,
                 mask: torch.Tensor) -> List[Tuple[List[int], float]]:
        """
        Get the most likely tags for the given inputs.
        """
        _, max_seq_length, num_tags = unary_potentials.size()

        # Get the tensors out of the variables
        unary_potentials, mask = unary_potentials.data, mask.data

        start_tag = num_tags
        end_tag = num_tags + 1

        aug_binary_potentials = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-10000.)

        # set the current transition matrix to the learned transitions
        aug_binary_potentials[:num_tags, :num_tags] = binary_potentials

        # Since we're not learning start and end transitions, we will set
        # them all to the same low potential.
        aug_binary_potentials[start_tag, :num_tags] = -10000.0
        aug_binary_potentials[:num_tags, end_tag] = -10000.0

        best_paths = []
        # Pad the max sequence length by 2 to account for start_tag + end_tag.
        tag_sequence = torch.Tensor(max_seq_length + 2, num_tags + 2)

        for prediction, prediction_mask in zip(unary_potentials, mask):
            sequence_length = torch.sum(prediction_mask)

            # Start with everything totally unlikely
            tag_sequence.fill_(-10000.)
            # At timestep 0 we must have the START_TAG
            tag_sequence[0, start_tag] = 0.
            # At steps 1, ..., sequence_length we just use the incoming
            # prediction
            tag_sequence[1:(sequence_length + 1), :num_tags] = \
                    prediction[:sequence_length]
            # And at the last timestep we must have the END_TAG
            tag_sequence[sequence_length + 1, end_tag] = 0.

            # We pass the tags and the transitions to ``decode``.
            path, score = \
                    self.decode(tag_sequence[:(sequence_length + 2)],
                                aug_binary_potentials)
            # Get rid of START and END sentinels and append.
            path = path[1:-1]
            best_paths.append((path, score.item()))

        return best_paths

    def decode(self,
               unary_potentials: torch.Tensor,
               binary_potentials: torch.Tensor):
        """
        Perform decoding in log space over a sequence given a matrix specifying
        unary potentials for possible tags per timestep and a transition matrix
        specifying pairwise (transition) potentials between tags.

        This is where you should implement the decoding algorithm you derived
        in A4 section 1.

        Parameters
        ----------
        unary_potentials : torch.Tensor, required.
            A tensor of shape (sequence_length, num_tags) representing unary
            potentials for a set of tags over a given sequence.
        binary_potentials : torch.Tensor, required.
            A tensor of shape (num_tags, num_tags) representing the binary
            potentials for transitioning between a given pair of tags.

        Returns
        -------
        path : List[int]
            The tag indices of the maximum likelihood tag sequence.
        score : torch.Tensor
            The score of the path.
        """

        seq_length, num_tags = unary_potentials.size()
        heart = [unary_potentials[0, :]]
        back = []

        for i in range(1, seq_length):
            score = heart[i-1].unsqueeze(-1) + binary_potentials + unary_potentials[i, :]
            max_score, pointers = torch.max(score, dim = 0)
            heart.append(max_score)
            back.append(pointers)

        final_score, pointer = torch.max(heart[-1], 0)
        tag_indices = [int(pointer)]

        for backPointer in reversed(back):
            pointing_to = tag_indices[-1]
            tag_indices.append(int(backPointer[pointing_to]))

        tag_indices.reverse()
        return tag_indices, final_score
