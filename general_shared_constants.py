class RelPosEmbsChoices:
    no_rel_pos_embs = "no_rel_pos_embs"
    one_embedder = "one_embedder"
    two_embedders = "two_embedders"

    __choices__ = {
        one_embedder, 
        two_embedders, 
        no_rel_pos_embs,
    }


class AbsPosEmbsModes:
    no_abs_pos_embs = "no_abs_pos_embs"
    fixed_pos_embs = "fixed_pos_embs"
    learned_pos_embs = "learned_pos_embs"
    
    __choices__ = {
        no_abs_pos_embs, 
        fixed_pos_embs,
        learned_pos_embs,
    }


class DatasetTypesChoices:
    most_basic_dataset = "most_basic_dataset"
    oracle_basic_dataset = "oracle_basic_dataset"
    self_learned_basic_dataset = "self_learned_basic_dataset"

    __choices__ = {
        most_basic_dataset,
        oracle_basic_dataset,
        self_learned_basic_dataset,
    }
