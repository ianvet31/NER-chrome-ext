from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.http import JsonResponse
from django.http import HttpResponseNotAllowed

import json
import nltk

import spacy

from django.views.decorators.csrf import csrf_exempt



def index(request):
    return HttpResponse("Hello, world. You're at the ner index.")


@csrf_exempt
def ner(request):
    if request.method == 'POST':
        text = json.loads(request.body)['text']

        if text is None:
            return JsonResponse({'error': 'Missing "text" parameter'})

        # Perform NER using nltk
        #sentences = nltk.sent_tokenize(text)
        #tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
        #tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
        #print(text)

        nlp = spacy.load("en_core_web_sm")

        doc = nlp(text)
        named_entities = []
        ne_labels = []
        for ent in doc.ents:
            named_entities.append(ent.text)
            ne_labels.append(ent.label_)
        response = JsonResponse({'ents': named_entities, 'labels': ne_labels}, safe=False)

        

        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'

        return response

    # return a 405 Method Not Allowed response for non-POST requests
    return HttpResponseNotAllowed(['POST'])
