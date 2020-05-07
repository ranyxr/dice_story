"""
Keywords to Knowledge graph
"""
from py2neo import Graph
from itertools import combinations


graph = Graph(password='root')

keywords = ['love', 'drink', 'cola', 'beer', 'cat']
cues_pos = {'v', 'a'}  # verb, adj
entities_pos = {'n'}  # noun

# Build query
is_related = """
MATCH (start {name:"%s"})-[r]-(end {name:"%s"})
RETURN r
"""

check_pos = """
match (n:Lemma {name:"%s"})
return n.pos
"""

add_cue_or_entity = """
MATCH (n {name:"%s"})-[r]-(m:Lemma)
WHERE m.pos in %s and size(m.name)>1
WITH m, rand() AS number
RETURN m.name
ORDER BY number
LIMIT %d
"""

entity_relation = {}

# Check relation between keywords
keyword_pairs = [pair for pair in combinations(keywords, 2)]
for pair in keyword_pairs:
    table = graph.run(is_related % pair).to_table()
    if len(table):
        keyword1, keyword2 = pair
        pos1 = graph.run(check_pos % keyword1).to_table()
        pos2 = graph.run(check_pos % keyword2).to_table()

        if {p[0] for p in list(pos1)} & cues_pos:
            if entity_relation.get(keyword1):
                entity_relation[keyword1].add(keyword2)
            else:
                entity_relation[keyword1] = {keyword2}

        elif {p[0] for p in list(pos2)} & cues_pos:
            if entity_relation.get(keyword2):
                entity_relation[keyword2].add(keyword1)
            else:
                entity_relation[keyword2] = {keyword1}

# Add relation to un-referred keywords
values = set()
referred_keywords = {key for key in entity_relation.keys()} | \
                    {values.update(value) for value in entity_relation.values()}

unreferred_keywords = set(keywords) - referred_keywords
for word in unreferred_keywords:
    pos = graph.run(check_pos % word).to_table()
    if {p[0] for p in list(pos)} & cues_pos:
        # add entity
        new_entity = graph.run(add_cue_or_entity % (word, list(entities_pos), 3)).to_table()
        entity_relation[word] = {entity[0] for entity in list(new_entity)}

    elif {p[0] for p in list(pos)} & entities_pos:
        # add cues
        new_cue = graph.run(add_cue_or_entity % (word, list(cues_pos), 2)).to_table()
        if not len(new_cue):
            continue
        cue = list(new_cue)[0][0]
        if entity_relation.get(cue):
            entity_relation[cue].add(word)
        else:
            entity_relation[cue] = {word}

print(entity_relation)
