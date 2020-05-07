import sys
import ray
sys.path.append('../gpt-2-keyword-generation')
from keyword_encode import encode_keywords


ray.init(object_store_memory=100 * 1000000,
        redis_max_memory=100 * 1000000)

encode_keywords(csv_path='../data/cleaned_rocstories.csv',
                out_path='../data/encoded_rocstories_with_relation.txt',
                body_field='content',
                title_field='storytitle',
                keyword_gen='content',
                max_keywords=7,
                min_keywords=3,
                )
