def pre_process(file):
    with open(f'./json/{file}', 'r', encoding='utf-8') as f, open(f'./processed_data/{file}', 'w', encoding='utf=8') as w:
        for line in tqdm(f.readlines()):
            line = json.loads(line)
            keys = ['ner', 'relations', 'sentences']
            lengths = [len(line[k]) for k in keys]
            assert len(set(lengths)) == 1
            sentence_lengths = [len(s) for s in line["sentences"]]
            sentence_starts = np.cumsum(sentence_lengths)
            sentence_starts = np.roll(sentence_starts, 1)
            for i in range(lengths[0]):
                dic = {'text': ' '.join(line['sentences'][i]),
                       'spo_list': []}
                for sh, st, oh, ot, l in line['relations'][i]:
                    sentence = line['sentences'][i]
                    start = sentence_starts[i]
                    sub = ' '.join(sentence[sh-start:st-start+1])
                    obj = ' '.join(sentence[oh-start:ot-start+1])
                    for ner in line["ner"][i]:
                        if ner[0] == sh and ner[1] == st:
                            sub_type = ner[2]
                        if ner[0] == oh and ner[1] == ot:
                            obj_type = ner[2]
                    dic['spo_list'].append({'predicate': l, 'subject': sub, 'subject_type': sub_type, 'object': obj, 'object_type': obj_type})
                if dic['spo_list'] != []:
                    w.write(json.dumps(dic,ensure_ascii=False) + '\n')
                    
                    
pre_process('train.json')
pre_process('dev.json')
pre_process('test.json')