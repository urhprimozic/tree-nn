# tree-nn
Tree Neural Networks with decision units

# Ideja 
izmenjaje uporabljamo Tree layer in nek drug NN layer (recimo fully connected)..

## Training
- lahko izmenjaje treniramo `DecisionUnit` in `NN`
- oboje skupaj
- rabmo nekej nedeterminističnosti in samplinga 
### Loss
- $\mathcal L(\text{path}(x)) = \text{ERROR}(M(x)) + |\text{zeros in hidden layers}(\text{path}(x))|$
- namesto #ničel lahko gledamo tudi kaj drugega (želimo recimo lahko, da dobimo iskalno drevo - če veš, da si pravokoten na $z$, mogoče ni smiselno, da je $<z, y>$ velik za $y$, ki se pojavi naprej v modelu...)