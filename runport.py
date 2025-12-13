# -*- coding: utf-8 -*-
import text_metrics
import sys
import json
import datetime
import time
import os

feat_list = ["adj_arg_ovl", "arg_ovl", "adj_stem_ovl", "stem_ovl", "adj_cw_ovl", "adjacent_refs", "anaphoric_refs", "if_ratio", "or_ratio", "and_ratio", "logic_operators", "negation_ratio", "hypernyms_verbs", "conn_ratio", "add_pos_conn_ratio", "add_neg_conn_ratio", "tmp_pos_conn_ratio", "tmp_neg_conn_ratio", "cau_pos_conn_ratio", "cau_neg_conn_ratio", "log_pos_conn_ratio", "log_neg_conn_ratio", "function_words", "content_words", "pronoun_ratio", "adjective_ratio", "words", "paragraphs", "verbs", "adverbs", "words_per_sentence", "syllables_per_content_word", "sentences_per_paragraph", "noun_ratio", "sentences", "flesch", "ttr", "personal_pronouns", "verbs_ambiguity", "adjectives_ambiguity", "nouns_ambiguity", "adverbs_ambiguity", "mean_noun_phrase", "words_before_main_verb", "min_cw_freq", "cw_freq"]

text = sys.argv[1]
use_json = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].lower() in ['true', '1', 'yes'] else False
raw = text.replace('{{quotes}}', '"')
raw = raw.replace('{{exclamation}}', '!')
raw = raw.replace('{{enter}}', '\n')
raw = raw.replace('{{sharp}}', '#')
raw = raw.replace('{{ampersand}}', '&')
raw = raw.replace('{{percent}}', '%')
raw = raw.replace('{{dollar}}', '$')

#print(raw)
raw = raw.encode("utf-8", "surrogateescape").decode("utf-8")
t = text_metrics.Text(raw)
ret = text_metrics.CMP_METRICS.values_for_text(t).as_flat_dict()

if use_json:
    
    processing_time = time.time()
    result_data = {
        "text": text,
        "timestamp": datetime.datetime.now().isoformat(),
        "metrics": {f.replace(" ", "_"): round(ret[f], 5) for f in feat_list}
    }
    result_json = json.dumps(result_data, indent=2, ensure_ascii=False)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/nilcmetrix_result_{timestamp}.json"

    os.makedirs('results', exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as file:
        file.write(result_json)
    print(f"Resultado salvo em: {filename}")
    print(result_json)
else:
    result = '' 
    for f in feat_list:
        m = "%s:%s," % (f.replace(" ", "_"), round(ret[f],5))
        #print(m)
        result += m
    print("++", result, "++")
