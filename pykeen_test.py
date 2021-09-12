from typing import List

import pykeen.nn
from pykeen.pipeline import pipeline
import pykeen
from scipy.spatial.distance import cosine

result = pipeline(model='TransE', dataset='UMLS')
model = result.model

# entity_representation_modules: List['pykeen.nn.RepresentationModule'] = model.entity_representations
# relation_representation_modules: List['pykeen.nn.RepresentationModule'] = model.relation_representations
#
# entity_embeddings: pykeen.nn.Embedding = entity_representation_modules[0]
# relation_embeddings: pykeen.nn.Embedding = relation_representation_modules[0]
#
# entity_embedding_tensor: torch.FloatTensor = entity_embeddings(indices=None)
# relation_embedding_tensor: torch.FloatTensor = relation_embeddings(indices=None)

entity_embedding_tensor = model.entity_representations[0](indices=None).cpu().detach().numpy()
print(entity_embedding_tensor.shape)

print(result.training.mapped_triples)
print(result.training.entity_ids)
print(result.training.extra_repr())


# for k, v in result.training.entity_to_id.items():
#     print(k, v)
#     print(f"tensor: {entity_embedding_tensor[v]}")


def get_similarity(ent_1, ent_2):
    cos = cosine(entity_embedding_tensor[result.training.entity_to_id[ent_1]],
                 entity_embedding_tensor[result.training.entity_to_id[ent_2]])
    print(f"Similarity of {ent_1} and {ent_2} is {cos}")


get_similarity("cell", "human")
get_similarity("cell", "tissue")
get_similarity("cell", "fish")
get_similarity("cell", "lipid")

get_similarity("cell", "reptile")
get_similarity("cell", "behavior")
get_similarity("cell", "bird")
get_similarity("cell", "age_group")
