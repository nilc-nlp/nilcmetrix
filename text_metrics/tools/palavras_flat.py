from __future__ import unicode_literals, print_function, division
from text_metrics.conf import config
# import subprocess
# import tempfile
# import os
# import codecs
from urllib.parse import urlencode
from urllib.request import urlopen


# def palavras_flat(t):
#     fdesc, input_file_path = tempfile.mkstemp(text=True)
#     os.close(fdesc)
#     with codecs.open(input_file_path, mode='w', encoding='utf-8') as infile:
#             infile.write(t.raw_content)
#     command = 'php {palavras} {file}'.format(
#         palavras=config['CALL_PALAVRAS_FLAT'],
#         file=input_file_path,
#     )
#     palavras_tree = subprocess.check_output(command, shell=True)
#     os.remove(input_file_path)
#     return str(palavras_tree)


# def palavras_flat(t):
#     '''
#     Call a webservice to run the parser Palavras
#
#     :param text: the text to be parsed, in unicode.
#     :return: the response string from Palavras
#     '''
#     params = {'sentence': t.raw_content}
#     data = urlencode(params).encode('utf-8')
#     f = urlopen(config['CALL_PALAVRAS_FLAT'], data)
#     response = f.read()
#     f.close()
#
#     return str(response)

def palavras_flat(t):
    '''
    Call a webservice to run the parser Palavras

    :param text: the text to be parsed, in unicode.
    :return: the response string from Palavras
    '''
    params = {'sentence': t.raw_content}
    data = urlencode(params).encode('utf-8')
    f = urlopen(config['CALL_PALAVRAS_FLAT'], data)
    response = f.read()
    f.close()

    return response.decode('utf-8') \
                   .replace('\\n', '\n') \
                   .replace('\\t', '\t') \
                   .replace('ß', 's')

# def palavras_flat(t):
#     result = ""
#     print("RESULT FROM PALAVRAS: ")
#
#     try:
#         #need to be at the same machine where palavras is installed
#         #default path to palavras /opt/palavras/
#
#         value = ''
#         if isinstance(t, str):
#            value = t
#         else:
#            value = t.raw_content
#
#         command = 'echo "{text}" | /opt/palavras/por.pl --dep-retokenize'.format(text=value)
#         result = subprocess.check_output(command, shell=True)
#         print(result)
#
#         return result.decode('unicode-escape')\
#             .replace(u"<\xc3\x9f>", "<s>")\
#             .replace(u"</\xc3\x9f>", "</s>")\
#             .encode('latin1')\
#             .decode('utf-8')
#     except Exception as ex:
#         print("ERROR PALAVRAS-FLAT:")
#         print(ex)
#         return result
