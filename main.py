from music21 import converter, instrument, note, chord
import torch
import os

data = []
targets = []
notes = []

batch_size = 5
sequence_length = 50

for file in os.listdir("./data"):
    mid_file = converter.parse("./data/" + file)

    tokens = None
    parts = instrument.partitionByInstrument(mid_file)
    if parts:
        tokens = parts.parts[0].recurse()
    else:
        tokens = mid_file.flat.notes

    for token in tokens:
        if isinstance(token, note.Note):
            notes.append(str(token.pitch))
        elif isinstance(token, chord.Chord):
            notes.append(".".join(str(note) for note in token.normalOrder))

pitches = sorted(set(notes))
note_map = dict((pitch, i) for i, pitch in enumerate(pitches))

for i in range(len(notes) - sequence_length):
    curr = notes[i:i + sequence_length]
    target = notes[i + sequence_length]
    data.append([note_map[note] for note in curr])
    targets.append(target)

data = torch.tensor(data)
data = torch.reshape(data, (len(data), sequence_length, 1))
data = torch.div(data, len(pitches))
print(data)
