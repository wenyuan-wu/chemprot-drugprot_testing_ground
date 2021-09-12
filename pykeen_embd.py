from pykeen.pipeline import pipeline
from pykeen.datasets import OGBBioKG, UMLS, OpenBioLink
import pykeen.datasets.analysis as pda
# from ogb.linkproppred import LinkPropPredDataset

result = pipeline(model='TransE', dataset=OGBBioKG())
# dataset = LinkPropPredDataset(name="ogbl-biokg")
# biokg = OGBBioKG()
# print(biokg)
# print(biokg.training.entity_to_id)
# print(pda.get_entity_count_df("UMLS"))
