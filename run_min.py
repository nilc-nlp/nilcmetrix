# -*- coding: utf-8 -*-
import text_metrics
import sys
import time
import json
import os
from datetime import datetime

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

# calcular tempo de processamento
start_time = time.time()
ret = text_metrics.no_palavras_metrics.values_for_text(t).as_flat_dict()
end_time = time.time()

if use_json:

    processing_time = time.time() - start_time

    result_data = {
        "text": text,
        "timestamp": datetime.now().isoformat(),
        "processing_time_seconds": round(processing_time, 3),
        "metrics": ret
    }

    result_json = json.dumps(result_data, indent=2, ensure_ascii=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/nilcmetrix_result_{timestamp}.json"
    filepath = filename
    
    os.makedirs('results', exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(result_json)

    print(f"Resultado salvo em: {filepath}")
    print(result_json)

else:
    result = '' 
    for f in ret:
        m = "%s:%s," % (f, ret[f])
        #print(m)
        result += m
    print("++", result, "++")



