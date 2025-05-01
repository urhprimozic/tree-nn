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
- mogoče hočmo preprečit, da bi na več različnih koncih testiral iste stvari. Zato bi lah gledal, da v različnih poteh ni istih vprašanj... Ampak ta ideja je mogoče premočna za drevesno strukturo. Če bi neki bl zakompliciral, bi lah nardil en model, ki se efekitvno "zapolne neko dejstvo" o inputu, in ga ne rab večrkat računat... Ampak sj to kao vsi modeli delajo.

- **Kako prepoznati parametre, ki "ne vplivajo" na rezultat:** Težko. Lahko prepoznaš, ali sprememba parametra vpliva na rezultat (=odvod), ampak nočno čist tega. Če pa je odvod po temu parametru mali za večino inputov, pol pa je slabo. Tok da lah bi gledal neko "average velikost odvoda", pa hotu, da ni nič...? Ker če je odvod $\frac{dM(x)}{dp} = 0$ za zelo veliko $x \in X$, je mogoče tvoja funkcija v $p$ konstantna povsod....  Tega se lotimo z  REMSG in EMSG