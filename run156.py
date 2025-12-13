# -*- coding: utf-8 -*-
import text_metrics
import sys
import json
import datetime
import time
import os

feat_list = ["sentence_length_max","words","words_per_sentence","brunet","ratio_coordinate_conjunctions","gunning_fox","sentence_length_min","adjectives_min","punctuation_diversity","adjectives_max","dep_distance","flesch","long_sentence_ratio","sentences_with_five_clauses","gerund_verbs","verbs","short_sentence_ratio","honore","medium_long_sentence_ratio","yngve","coordinate_conjunctions_per_clauses","idade_aquisicao_1_25_ratio","indicative_imperfect_ratio","concretude_mean","subjunctive_present_ratio","prepositions_per_sentence","logic_operators","third_person_pronouns","relative_pronouns_ratio","ttr","aux_plus_PCP_per_sentence","dalechall_adapted","tmp_pos_conn_ratio","ratio_subordinate_conjunctions","pronouns_max","pronoun_ratio","tmp_neg_conn_ratio","sentences_with_six_clauses","log_pos_conn_ratio","abstract_nouns_ratio","adverbs_ambiguity","frazier","apposition_per_clause","adjective_ratio","adjectives_ambiguity","sentences_with_seven_more_clauses","sentences_with_four_clauses","subjunctive_imperfect_ratio","imageabilidade_25_4_ratio","preposition_diversity","min_cw_freq","subordinate_clauses","adverbs_diversity_ratio","idade_aquisicao_std","inflected_verbs","easy_conjunctions_ratio","first_person_pronouns","familiaridade_4_55_ratio","if_ratio","familiaridade_mean","syllables_per_content_word","postponed_subject_ratio","add_pos_conn_ratio","sentences_with_two_clauses","infinite_subordinate_clauses","concretude_1_25_ratio","indicative_preterite_perfect_ratio","hypernyms_verbs","idade_aquisicao_mean","max_noun_phrase","adverbs","concretude_std","nouns_ambiguity","idade_aquisicao_55_7_ratio","passive_ratio","third_person_possessive_pronouns","oblique_pronouns_ratio","imageabilidade_55_7_ratio","verb_diversity","subjunctive_future_ratio","simple_word_ratio","or_ratio","content_density","second_person_pronouns","familiaridade_1_25_ratio","indefinite_pronoun_ratio","cau_pos_conn_ratio","relative_pronouns_diversity_ratio","conn_ratio","add_neg_conn_ratio","first_person_possessive_pronouns","imageabilidade_std","indicative_present_ratio","imageabilidade_mean","indicative_pluperfect_ratio","concretude_55_7_ratio","function_word_diversity","and_ratio","pronoun_diversity","verbs_max","non-inflected_verbs","content_words","verbal_time_moods_diversity","personal_pronouns","adverbs_before_main_verb_ratio","familiaridade_std","adverbs_min","adjunct_per_clause","medium_short_sentence_ratio","infinitive_verbs","cau_neg_conn_ratio","sentences_with_zero_clause","adjective_diversity_ratio","content_word_diversity","verbs_ambiguity","idade_aquisicao_25_4_ratio","nouns_min","log_neg_conn_ratio","cw_freq","nouns_max","adverbs_max","familiaridade_25_4_ratio","sentences_with_three_clauses","named_entity_ratio_sentence","familiaridade_55_7_ratio","content_word_min","relative_clauses","indefinite_pronouns_diversity","non_svo_ratio","imageabilidade_4_55_ratio","ratio_function_to_content_words","clauses_per_sentence","temporal_adjunct_ratio","idade_aquisicao_4_55_ratio","concretude_4_55_ratio","min_noun_phrase","words_before_main_verb","content_word_max","named_entity_ratio_text","dialog_pronoun_ratio","punctuation_ratio","mean_noun_phrase","std_noun_phrase","function_words","pronouns_min","negation_ratio","noun_diversity","verbs_min","prepositions_per_clause","participle_verbs","concretude_25_4_ratio","indicative_condition_ratio","sentences_with_one_clause","noun_ratio","content_words_ambiguity","hard_conjunctions_ratio"]

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
ret = text_metrics.all_metrics.values_for_text(t).as_flat_dict()

if use_json:
    processing_time = time.time()
    result_data = {
        "text": text,
        "timestamp": datetime.datetime.now().isoformat(),
        "metrics": {f: round(ret[f], 4) for f in feat_list}
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
    result_array = []
    for f in feat_list:
        result_array.append(round(ret[f],4))
        #m = "%s:%s," % (f.replace(" ", "_"), round(ret[f],5))
        #print(m)
        #result += m
    result = str(result_array)
    result = result.replace(" ", "")
    print(result)
