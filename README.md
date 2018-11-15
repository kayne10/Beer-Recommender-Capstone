# Beer Recommender

## Summary

## EDA


## LDA Results

(Elbow Plot?)

| Topic Number     | Topics | Possible Category |
| ---      | ---       |
| 0 |'ale' 'pale' 'apa' 'amber' 'red' 'blonde' 'brown' 'wheat' 'saison' 'farmhouse'                 | APAs |
| 1 |'ipa' 'double' 'imperial' 'hop' 'esb' 'bitter' 'extra' 'strong' 'belgian' 'session'                | IPAs |
| 2 |'porter' 'baltic' 'coconut' 'doppelbock' 'vanilla' 'robust' 'plum' 'chocolate' 'java' 'voodoo'| Porters |
| 3 |'pilsner' 'pilsener' 'german' 'czech' 'pils' 'barrel' 'grand' 'brew' 'golden' 'canyon'| Pilsners |
| 4 |'white' 'witbier' 'märzen' 'wit' 'lager' 'house' 'king' 'street' 'bier' 'cold'| ? |
| 5 |'belgian' 'stout' 'helles' 'oatmeal' 'black' 'light' 'munich' 'vienna' 'adjunct' 'milk'| Stouts|
| 6 |'kölsch' 'winter' 'warmer' 'tripel' 'berliner' 'weissbier' 'island''shandy' 'summer' 'kentucky'| ? |
| 7 |'beer' 'fruit' 'vegetable' 'rye' 'cream' 'pumpkin' 'wheat' 'great''spiced' 'herbed' | ? |
| 8 |'cider' 'altbier' 'gold' 'apple' 'hard' 'dry' 'jack' 'ginger' 'miner' 'angry'| Ciders |

## Input/Output Vectors

| Topic probabilities | ABU | IBU|
| --- | ---| --- |
| | | |
|... |... |... |

## Example

Recommendation of Beer within Dataset
```
{'Hard Apple': ['Ginger Cider', "Wolfman's Berliner", 'Schilling Hard Cider', 'Hard Cider', 'Nomader Weiss', 'Monkey Chased the Weasel', 'Magic Apple', 'Contemplation', 'Totally Radler']}
```

Recommendation for User's preference
```
{'target_beer': 'Voodoo Ranger Imperial IPA', 'recommendations': ['Long Hammer IPA', 'Long Hammer IPA', 'Colorado Red Ale', 'Manzanita Pale Ale', 'Pine Belt Pale Ale', 'Rebel IPA', 'Fremont Summer Ale', 'Pine Belt Pale Ale', 'Belgorado', 'Pretzel Stout']}
```

## Demo
