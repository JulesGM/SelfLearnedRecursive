# Self Learned Explanations for Transformer Language Models
- 



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
How to make it faster:
 - No beam search
 - Better work stack building
 - Modify generation lengths
 - We tok encode and decode for kind of no reason
 - Multiple levels at once



- [Not urgent] If we want to do multi-gpu, the code for pred_fn of self learned scratchpads needs to be fixed.
- [Is something stupid] Fix the double <bos> thing maybe. Why woudl the model EVER predict <bos>? It should never learn to do that.


FUTURE:
- Find a new Dataset (some DeepChem thing maybe)
- use pre-trained BART