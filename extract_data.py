import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import re


# for f in tqdm(os.listdir('data'), total = len(os.listdir('data'))):
#     with open('data/{}'.format(f)) as js:
#         try:
#             dat = json.load(js)
#         except UnicodeDecodeError: continue
#         try:
#             if dat['hasResults'] and 'resultsSection' in dat.keys() and 'analyses' in dat['resultsSection']['outcomeMeasuresModule']['outcomeMeasures'][0].keys() and \
#                 'phases' in dat['protocolSection']['designModule'].keys() and ('PHASE3' in dat['protocolSection']['designModule']['phases'] or 'PHASE2' in dat['protocolSection']['designModule']['phases']):
#                     extracted_dat = dict()
#                     extracted_dat['Organization'] = dat['protocolSection']['identificationModule']['organization']['class']
#                     if 'collaborators' in dat['protocolSection']['sponsorCollaboratorsModule']:
#                         extracted_dat['Collaborators'] = [c['name'] for c in dat['protocolSection']['sponsorCollaboratorsModule']['collaborators']]
#                     else:
#                         extracted_dat['Collaborators'] = []
#                     extracted_dat['Conditions'] = dat['protocolSection']['conditionsModule']['conditions']
#                     extracted_dat['Study_Type'] = dat['protocolSection']['designModule']['studyType']
#                     if extracted_dat['Study_Type'] != 'INTERVENTIONAL':
#                         continue
#                     if 'armGroups' not in dat['protocolSection']['armsInterventionsModule'].keys():
#                         continue
#                     extracted_dat['Arms Info'] = dat['protocolSection']['armsInterventionsModule']['armGroups']
#                     extracted_dat['Phase'] = dat['protocolSection']['designModule']['phases']
#                     if 'designInfo' not in dat['protocolSection']['designModule'].keys():
#                         continue
#                     extracted_dat['Design Info'] = dat['protocolSection']['designModule']['designInfo']
#                     extracted_dat['Enrollment'] = dat['protocolSection']['designModule']['enrollmentInfo']['count']
#                     extracted_dat['Eligibility'] = dat['protocolSection']['eligibilityModule']
#                     extracted_dat['Sponsor'] = dat['protocolSection']['sponsorCollaboratorsModule']['leadSponsor']
#                     if 'contactsLocationsModule' in dat['protocolSection'].keys() and 'locations' in dat['protocolSection']['contactsLocationsModule'].keys():
#                         extracted_dat['Locations'] = dat['protocolSection']['contactsLocationsModule']['locations']
#                     if 'conditionBrowseModule' in dat['derivedSection'].keys() and 'browseLeaves' in dat['derivedSection']['conditionBrowseModule'].keys(): 
#                         extracted_dat['Derived'] = dat['derivedSection']['conditionBrowseModule']['browseLeaves']
#                     if 'interventionBrowseModule' in dat['derivedSection'].keys() and 'browseLeaves' in dat['derivedSection']['interventionBrowseModule'].keys(): 
#                         if 'Derived' in extracted_dat.keys():
#                             extracted_dat['Derived'] += dat['derivedSection']['interventionBrowseModule']['browseLeaves']
#                         else:
#                             extracted_dat['Derived'] = dat['derivedSection']['interventionBrowseModule']['browseLeaves']
                    
#                     extracted_dat['Results'] = list(filter(lambda t: t['type'] == 'PRIMARY', dat['resultsSection']['outcomeMeasuresModule']['outcomeMeasures']))
#                     # with open('{}'.format(f), 'w') as file:
#                     #     json.dump(dat, file)
#                     # print(f)
#                     with open('extracted_data/{}'.format(f), 'w') as file:
#                         json.dump(extracted_dat, file)
#         except Exception as e:
#             print(e)
#             with open('{}'.format(f), 'w') as file:
#                 json.dump(dat, file)
#             print(f)
#             break

i = 0
ls = []
ph2_dict = dict()
for f in tqdm(os.listdir('extracted_data'), total = len(os.listdir('extracted_data'))):
    with open('extracted_data/{}'.format(f)) as js:
        dat = json.load(js)
        if 'analyses' not in dat['Results'][0].keys() or 'pValue' not in dat['Results'][0]['analyses'][0].keys(): continue
        if 'PHASE2' in dat['Phase'] and 'PHASE3' not in dat['Phase']:
            s = set()

            for d in dat['Arms Info']:
                if 'interventionNames' in d.keys():
                    if d['type'] == 'EXPERIMENTAL':
                        s = s | set(d['interventionNames'])

            for d in dat['Arms Info']:
                if 'interventionNames' in d.keys():
                    if d['type'] != 'EXPERIMENTAL':
                        s = s - set(d['interventionNames'])
            for inter in s:
                intervention = inter.split(':')[1][1:].lower()
                if intervention in ph2_dict.keys():
                    ph2_dict[intervention].append(([(d['id'], d['relevance']) for d in dat['Derived']] if 'Derived' in dat.keys() else '',[dat['Results'][0]['analyses'][0]['pValue']]))
                else:
                    ph2_dict[intervention] = [([(d['id'], d['relevance']) for d in dat['Derived']] if 'Derived' in dat.keys() else '',[dat['Results'][0]['analyses'][0]['pValue']])]
                interventions = inter.split(':')[1][1:].lower().split(' ')
                for intervention in interventions:
                    if intervention in ph2_dict.keys():
                        ph2_dict[intervention].append(([(d['id'], d['relevance']) for d in dat['Derived']] if 'Derived' in dat.keys() else '',[dat['Results'][0]['analyses'][0]['pValue']]))
                    else:
                        ph2_dict[intervention] = [([(d['id'], d['relevance']) for d in dat['Derived']] if 'Derived' in dat.keys() else '',[dat['Results'][0]['analyses'][0]['pValue']])]
print(len(ph2_dict.keys()))          
for f in tqdm(os.listdir('extracted_data'), total = len(os.listdir('extracted_data'))):
    with open('extracted_data/{}'.format(f)) as js:
        dat = json.load(js)
        if 'PHASE3' not in dat['Phase']: continue
        if 'analyses' not in dat['Results'][0].keys() or 'pValue' not in dat['Results'][0]['analyses'][0].keys() or 'maskingInfo' not in dat['Design Info'].keys(): continue
        if dat['Design Info']['maskingInfo']['masking'] == 'NONE': continue

        ls.append([])
        ls[i].append(dat['Conditions'])
        ls[i].append(dat['Organization'])
        ls[i].append(dat['Collaborators'])
        ls[i].append(dat['Design Info']['allocation'] if 'allocation' in dat['Design Info'].keys() else 'RANDOMIZED')
        ls[i].append(dat['Design Info']['interventionModel'] if 'interventionModel' in dat['Design Info'].keys() else 'SINGLE')
        ls[i].append(dat['Design Info']['primaryPurpose'] if 'primaryPurpose' in dat['Design Info'].keys() else '')
        ls[i].append(dat['Design Info']['maskingInfo']['masking'])
        ls[i].append(dat['Enrollment'])
        ls[i].append(dat['Eligibility']['healthyVolunteers'] if 'healthyVolunteers' in dat['Eligibility'].keys() else '')
        ls[i].append(dat['Eligibility']['sex'])
        if 'minimumAge' in dat['Eligibility'].keys():
            ls[i].append(dat['Eligibility']['minimumAge'])
        else:
            ls[i].append('0 Years')
        if 'maximumAge' in dat['Eligibility'].keys():
            ls[i].append(dat['Eligibility']['maximumAge'])
        else:
            ls[i].append('120 Years')
        ls[i].append(len(dat['Locations']) if 'Locations' in dat.keys() else 1)
        ls[i].append([(d['id'], d['relevance']) for d in dat['Derived']] if 'Derived' in dat.keys() else '')
        ls[i].append([(d['interventionNames'], d['type']) if 'interventionNames' in d.keys() else ([], d['type']) for d in dat['Arms Info'] ])


        ls[i].append([])
        exp_arm = False
        s = set()
        for d in dat['Arms Info']:
            if 'interventionNames' in d.keys():
                if d['type'] == 'EXPERIMENTAL':
                    exp_arm = True
                    s = s | set(d['interventionNames'])
        for d in dat['Arms Info']:
            if 'interventionNames' in d.keys():
                if d['type'] == 'EXPERIMENTAL': continue
                if not exp_arm and d['type'] == 'ACTIVE_COMPARATOR':
                    s = s | set(d['interventionNames'])
                else:
                    s = s - set(d['interventionNames'])
        for inter in s:
            intervention = inter.split(':')[1][1:].lower()
            if intervention in ph2_dict.keys():
                ls[i][-1]+=ph2_dict[intervention]
            else:
                interventions = re.findall(r'\(([^)]+)', intervention)
                interventions += re.findall(r'.*(?= \(?\d+[.]?\d+ ?mg)', intervention)
                interventions += intervention.split(' ')
                for ints in interventions:
                    if ints in ph2_dict.keys():
                        ls[i][-1]+=ph2_dict[ints]

        # else:
        #     if exp_arm:
        #         ls[i].append([])
        #     else:
        #         for d in dat['Arms Info']:
        #             if 'interventionNames' in d.keys():
        #                 if d['type'] == 'ACTIVE_COMPARATOR':
        #                     if d['interventionNames'][0].split(' ')[1].lower() in ph2_dict.keys() and 'placebo' not in d['interventionNames'][0].split(' ')[1].lower():
        #                         ls[i].append(ph2_dict[d['interventionNames'][0].split(' ')[1].lower()])
        #                         break
        #         else:
        #             ls[i].append([])


            
        ls[i].append(dat['Results'][0]['analyses'][0]['pValue'])
        i+=1

df = pd.DataFrame(ls, columns=['Conditions', 'Organization', 'Collaborators', 'Allocation', 'Intervention Model', 'Primary Purpose', 'Blinding', \
                               'Enrollment', 'Healthy Volunteers', 'Sex', 'Minimum Age', 'Maximum Age', 'Number Locations', \
                                'Derived terms', 'Interventions', 'PH2 Results', 'Results'])

df.to_csv('extracted_data.csv')