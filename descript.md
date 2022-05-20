# Self Learned Explanations for Transformer Language Models
Things to do:
x  - Fix generation batching
   x ---> modify the bart class to correct the decoder embeddings
   x ---> corrected decoder padding
   x ---> fixed caching
   x --- ---> Check that the outputs are identical with and without caching

x - Fix the length of the generations for the fake labels
x ---> max_gen = len(decoder_input_ids) + 

- Try oracle with just the parens
- Run experiments :)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
How to make it faster:
 - Only use parens. Would reduce generation length by a lot, & training speed as well even for oracle mode.
 - No beam search
 - Better work stack building
 - Modify generation lengths
 - We tok encode and decode for kind of no reason
 - Use curriculum learning & self learning, For a certain node, do both children at once?
 -> Do different, independant positions in the tree simultaneously?

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-> Mask prediction in label for training, for self training!
WHAT THIS MEANS: 
Training naturally has input_ids, labels and decoder_input_ids. we would need to have
labels that are -100 at the positions where we have self learned predictions.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
- See what's up with the different types of generation , for inference time.
---> If fully free beats the baseline, it's already good enough.
---> With help, it's better
---> Just results mode could be interesting.

- Review the effect of cleaning on self learned

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Curriculum learning
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~










- [Not urgent] If we want to do multi-gpu, the code for pred_fn of self learned scratchpads needs to be fixed.
- [Is something stupid] Fix the double <bos> thing maybe. Why woudl the model EVER predict <bos>? It should never learn to do that.

- Run experiments:
   [ ] - Hyperparameter search ? (Did some of that)
   [ ] - With no explanation vs oracle explanation vs Self-learned explanation (Did some of that)
   [ ] - No explanation vs curriculum self explanation vs curriculum no explanation vs curriculum oracle explanation 

FUTURE:
- Find a new Dataset (some DeepChem thing maybe)
- Scratchpad without the equation, just the result and the ops /or just the parens
- use pre-trained BART
- Attention mask


####################################
Two types of oracle scratchpads:
   - Oracle at training
   - Oracle at training and at inference  # That's nothing right

