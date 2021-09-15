"""
An approach to identify entities in a query. Uses a custom index for entity information.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""
import json
from tqdm import tqdm
from pathlib import Path
# from entity_linking.google_kg_api import get_entity_from_surface
from entity_linker.BERT_NER.bert import Ner
from entity_linker import surface_index_memory
from entity_linker.aaqu_entity_linker import IdentifiedEntity
from entity_linker.aaqu_util import normalize_entity_name, remove_prefixes_from_name, remove_suffixes_from_name

path = str(Path(__file__).parent.absolute())


class BertEntityLinker:

    def __init__(self, surface_index,
                 # Better name it max_entities_per_surface
                 max_entities_per_tokens=4,
                 model_path="/BERT_NER/out_base_gq/",
                 # model_path= "/BERT_NER/out_base_gq2_www",
                 # model_path= "/BERT_NER/out_base_gq1/",
                 # model_path= "/BERT_NER/out_base_web/",
                 device="cuda:0"
                 ):
        self.surface_index = surface_index
        self._model = Ner(path + model_path, device)

    def get_mentions(self, question: str):
        question = question.lower()
        output = self._model.predict(question)
        mentions = []
        current_mention = []
        for i, token in enumerate(output):
            if token['tag'][0] == 'B':
                current_mention.append(token['word'])
            elif token['tag'][0] == 'I':
                current_mention.append(token['word'])
            else:
                if len(current_mention) > 0:
                    mentions.append(' '.join(current_mention))
                current_mention = []
            if i == len(output) - 1 and len(current_mention) > 0:
                mentions.append(' '.join(current_mention))

        for i, mention in enumerate(mentions):
            # word_tokenize from nltk will change the left " to ``, which is pretty weird. Fix it here
            mentions[i] = mention.replace('``', '"').replace("''", '"')

        return mentions

    def _text_matches_main_name(self, entity, text):

        """
        Check if the entity name is a perfect match on the text.
        :param entity:
        :param text:
        :return:
        """
        text = normalize_entity_name(text)
        text = remove_prefixes_from_name(text)
        name = remove_suffixes_from_name(entity.name)
        name = normalize_entity_name(name)
        name = remove_prefixes_from_name(name)
        if name == text:
            return True
        return False

    def get_entities(self, utterance):
        entities = {}
        identified_entities = self.identify_entities(utterance)

        for entity in identified_entities:
            entities[entity.entity.id] = entity.entity.name

        return entities

    def identify_entities(self, utterance, min_surface_score=0.3):
        mentions = self.get_mentions(utterance)
        identified_entities = []
        mids = set()
        for mention in mentions:
            # use facc1
            entities = self.surface_index.get_entities_for_surface(mention)
            # use google kg api
            # entities = get_entity_from_surface(mention)
            # if len(entities) == 0:
            #     entities = get_entity_from_surface(mention)
            if len(entities) == 0 and len(mention) > 3 and mention.split()[0] == 'the':
                mention = mention[3:].strip()
                entities = self.surface_index.get_entities_for_surface(mention)
            elif len(entities) == 0 and f'the {mention}' in utterance:
                mention = f'the {mention}'
                entities = self.surface_index.get_entities_for_surface(mention)

            if len(entities) == 0:
                continue

            entities = sorted(entities, key=lambda x:x[1], reverse=True)
            for i, (e, surface_score) in enumerate(entities):
                if e.id in mids:
                    continue
                # Ignore entities with low surface score. But if even the top 1 entity is lower than the threshold,
                # we keep it
                if surface_score < min_surface_score and i > 0:
                    continue
                perfect_match = False
                # Check if the main name of the entity exactly matches the text.
                # I only use the label as surface, so the perfect match is always True
                if self._text_matches_main_name(e, mention):
                    perfect_match = True
                ie = IdentifiedEntity(mention,
                                      e.name, e, e.score, surface_score,
                                      perfect_match)
                # self.boost_entity_score(ie)
                identified_entities.append(ie)
                mids.add(e.id)

        identified_entities = sorted(identified_entities, key=lambda x: x.surface_score, reverse=True)

        return identified_entities


if __name__ == '__main__':
    # an example of how to use our BERT entity linker

    surface_index = surface_index_memory.EntitySurfaceIndexMemory(
        "data/entity_list_file_freebase_complete_all_mention", "data/surface_map_file_freebase_complete_all_mention",
        "../freebase_complete_all_mention")
    entity_linker = BertEntityLinker(surface_index, model_path="/BERT_NER/out_base_gq/", device="cuda:6")
    entity_linker.identify_entities("safety and tolerance of intermittent intravenous and oral zidovudine therapy in human immunodeficiency virus-infected pediatric patients. pediatric zidovudine phase i study group. is a medical trial for what?")

