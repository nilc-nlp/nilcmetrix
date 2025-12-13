# -*- coding: utf-8 -*-
import text_metrics
import sys
import json
import datetime
import time
import os

feat_list = ["adj_arg_ovl", "arg_ovl", "adj_stem_ovl", "stem_ovl", "adj_cw_ovl", "adjacent_refs", "anaphoric_refs", "if_ratio", "or_ratio", "and_ratio", "logic_operators", "negation_ratio", "hypernyms_verbs", "conn_ratio", "add_pos_conn_ratio", "add_neg_conn_ratio", "tmp_pos_conn_ratio", "tmp_neg_conn_ratio", "cau_pos_conn_ratio", "cau_neg_conn_ratio", "log_pos_conn_ratio", "log_neg_conn_ratio", "function_words", "content_words", "pronoun_ratio", "adjective_ratio", "words", "paragraphs", "verbs", "adverbs", "words_per_sentence", "syllables_per_content_word", "sentences_per_paragraph", "noun_ratio", "sentences", "flesch", "ttr", "personal_pronouns", "verbs_ambiguity", "adjectives_ambiguity", "nouns_ambiguity", "adverbs_ambiguity", "mean_noun_phrase", "words_before_main_verb", "min_cw_freq", "cw_freq"]

#feat_list = ["adjective_ratio", "adverbs", "content_words", "flesch", "function_words", "sentences_per_paragraph", "syllables_per_content_word", "words_per_sentence", "noun_ratio", "paragraphs", "sentences", "words", "pronoun_ratio", "verbs", "logic_operators", "and_ratio", "if_ratio", "or_ratio", "negation_ratio", "cw_freq", "min_cw_freq", "hypernyms_verbs", "brunet", "honore", "personal_pronouns", "ttr", "conn_ratio", "add_neg_conn_ratio", "add_pos_conn_ratio", "cau_neg_conn_ratio", "cau_pos_conn_ratio", "log_neg_conn_ratio", "log_pos_conn_ratio", "tmp_neg_conn_ratio", "tmp_pos_conn_ratio", "adjectives_ambiguity", "adverbs_ambiguity", "nouns_ambiguity", "verbs_ambiguity", "yngve", "frazier", "dep_distance", "content density", "words_before_main_verb", "adjacent_refs", "anaphoric_refs", "adj_arg_ovl", "arg_ovl", "adj_stem_ovl", "stem_ovl", "adj_cw_ovl", "adj_mean", "adj_std", "all_mean", "all_std", "paragraph_mean", "paragraph_std", "givenness_mean", "givenness_std", "span_mean", "span_std", "apposition_per_clause", "clauses_per_sentence", "prepositions_per_clause", "adjunct_per_clause", "prepositions_per_sentence", "relative_clauses", "aux_plus_PCP_per_sentence", "coordinate_conjunctions_per_clauses", "ratio_coordinate_conjunctions", "first_person_possessive_pronouns", "first_person_pronouns", "gerund_verbs", "infinitive_verbs", "inflected_verbs", "non-inflected_verbs", "participle_verbs", "passive_ratio", "second_person_possessive_pronouns", "second_person_pronouns", "sentences_with_five_clauses", "sentences_with_four_clauses", "sentences_with_one_clause", "sentences_with_seven_more_clauses", "sentences_with_six_clauses", "sentences_with_three_clauses", "sentences_with_two_clauses", "sentences_with_zero_clause", "simple_word_ratio", "ratio_subordinate_conjunctions", "third_person_possessive_pronouns", "third_person_pronouns", "adjective_diversity_ratio", "adjectives max", "adjectives min", "adjectives standard deviation", "adverbs_diversity_ratio", "adverbs max", "adverbs min", "adverbs standard deviation", "concretude_mean", "concretude_std", "concretude_1_25_ratio", "concretude_25_4_ratio", "concretude_4_55_ratio", "concretude_55_7_ratio", "content_word diversity", "content_word max", "content_word min", "content_word standard_deviation", "content_words_ambiguity", "dalechall adapted", "verbal_time_moods_diversity", "easy_conjunctions_ratio", "familiaridade_mean", "familiaridade_std", "familiaridade_1_25_ratio", "familiaridade_25_4_ratio", "familiaridade_4_55_ratio", "familiaridade_55_7_ratio", "function_word diversity", "gunning fox", "hard_conjunctions_ratio", "idade_aquisicao_mean", "idade_aquisicao_std", "idade_aquisicao_1_25_ratio", "idade_aquisicao_4_55_ratio", "idade_aquisicao_55_7_ratio", "idade_aquisicao_25_4_ratio", "imageabilidade_mean", "imageabilidade_std", "imageabilidade_1_25_ratio", "imageabilidade_25_4_ratio", "imageabilidade_4_55_ratio", "imageabilidade_55_7_ratio", "indefinite_pronouns_diversity", "medium_long_sentence_ratio", "max_noun_phrase", "mean_noun_phrase", "medium_short_sentence_ratio", "min_noun_phrase", "named_entity_ratio_sentence", "named_entity_ratio_text", "noun diversity", "nouns max", "nouns min", "nouns standard deviation", "subtitles", "postponed_subject_ratio", "preposition_diversity", "pronoun diversity", "pronouns max", "pronouns min", "pronouns standard deviation", "dialog_pronoun_ratio", "punctuation_diversity", "punctuation_ratio", "abstract_nouns_ratio", "adverbs_before_main_verb_ratio", "subjunctive_future_ratio", "indefinite_pronoun_ratio", "indicative_condition_ratio", "indicative_future_ratio", "indicative_imperfect_ratio", "indicative_pluperfect_ratio", "indicative_present_ratio", "indicative_preterite_perfect_ratio", "infinite_subordinate_clauses", "oblique_pronouns_ratio", "relative_pronouns_ratio", "subjunctive_imperfect_ratio", "subjunctive_present_ratio", "subordinate_clauses", "temporal_adjunct_ratio", "demonstrative_pronoun_ratio", "coreference_pronoum_ratio", "non_svo_ratio", "relative_pronouns_diversity_ratio", "sentence_length_max", "sentence_length_min", "sentence_length_standard_deviation", "short_sentence_ratio", "std_noun_phrase", "verb diversity", "verbs max", "verbs min", "verbs standard deviation", "long_sentence_ratio", "ratio_function_to_content_words"]

text = sys.argv[1]
use_json = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].lower() in ['true', '1', 'yes'] else False
raw = text.replace('{{quotes}}', '"')
raw = raw.replace('{{exclamation}}', '!')
raw = raw.replace('{{enter}}', '\n')
raw = raw.replace('{{sharp}}', '#')
raw = raw.replace('{{ampersand}}', '&')
raw = raw.replace('{{percent}}', '%')
raw = raw.replace('{{dollar}}', '$')

#raw = text.replace('\n', '').replace('\r', '')
#print(raw)
raw = raw.encode("utf-8", "surrogateescape").decode("utf-8")
t = text_metrics.Text(raw)
ret = text_metrics.CMP_METRICS.values_for_text(t).as_flat_dict()
#ret = text_metrics.all_metrics.values_for_text(t).as_flat_dict()

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
    print(result)
