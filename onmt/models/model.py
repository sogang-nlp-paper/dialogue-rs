""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch


class CLSModel(nn.Module):
    """
    Core trainable object in Classfication Task.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
    """

    def __init__(self, encoder, classifier):
        super(CLSModel, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, sent1, sent2):
        """Forward propagate a `sent1` and `sent2` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            sent1 (Tensor, LongTensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            sent2 (Tensor, LongTensor): A source sequence passed to encoder.

        Returns:

            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """

        # memory bank for attention (src_len, batch, hidden)
        enc_state, memory_bank1, lengths = self.encoder(sent1[0])
        enc_state, memory_bank2, lengths = self.encoder(sent2[0])

        # rnn max pooling
        sent1_emb = memory_bank1.max(dim=0)[0]
        sent2_emb = memory_bank2.max(dim=0)[0]

        # cnn max pooling
        # sent1_emb = memory_bank1.max(dim=-1)[0].transpose(0, 1)
        # sent2_emb = memory_bank2.max(dim=-1)[0].transpose(0, 1)
        cls_input = torch.cat([sent1_emb, sent2_emb,
                               torch.abs(sent1_emb - sent2_emb),
                               sent1_emb * sent2_emb], dim=1)
        out = self.classifier(cls_input)
        return out

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.classifier.update_dropout(dropout)


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.
    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.
        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
        Returns:
            (FloatTensor, dict[str, FloatTensor]):
            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder(src, lengths)

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
