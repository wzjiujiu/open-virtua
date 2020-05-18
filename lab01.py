import json
n=0

with open('states.json') as f:
    state_data=json.load(f)

print('original json keys:')
for state in state_data['states']:
    print(state_data['states'][n])
    n=n+1

for element in state_data['states']:
    element.pop('area_codes',None)
    print(element)

with open('new_states.json','w') as f:
    json.dump(state_data,f,indent=2)

with open('new_states.json') as f:
    state_data=json.load(f)

print('reloaded keys:')
n=0
for state in state_data['states']:
    print(state_data['states'][n])
    n=n+1



