# RAVE

## measuring

- is hard because output is not deterministic

## encoder

### weights

- rolling dimensions works well
    - may result in very stuttery sounds
- zeroing out entire tensor produces silence
- adding randomness in the order of 1e-2 works well
    - a lot of randomness will make everything stutter

```python
test = state_dict['encoder.net.0.weight']
test[:, 0, :] = torch.zeros_like(test[:, 0, :])
test[:, 0, :] = torch.zeros_like(test[:, 0, :])
test = torch.rand_like(test) * 1e-1
test += torch.rand_like(test) * 1e-2
test = torch.roll(test, 0, dims=2)
state_dict['encoder.net.0.weight'] = test
```

## decoder

- identify waveform conv, loudness conv and noise synth
    - maybe its in 'decoder.synth.branches.\[0, 1, 2\]' ?

### decoder.synth.branches.0

- possibly the waveform conv sub-network

#### weights

- replacing this generates strange timbres but with intact rhythm
- multiplying with a gain factor almost linearly changes the output gain!
- TODO: look at the structure? shape is `[16, 64, 7]`
    - in the code: `[out_dim, data_size * n_channels, 7]`

### decoder.synth.branches.1

- loudness conv sub-network

#### weights

- multiplying with a gain factor scales the envelope depth
    - negative gain creates strong pumping effects
- replacing with scalar creates randomly amplitude-modulated garbledness
    - negative values produce stronger dropouts/gatey-ness
- can actually be set to 0, creates very 'compressed'/dynamically flat
  output

#### bias

- kinda just does the same thing as the weights

### decoder.synth.branches.2

- possibly the noise synthesizer sub-network
- has 3 identical subnets
    - `.net.0`, `.net.2`, `.net.4`
- doesn't do a lot it seems?
