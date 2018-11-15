# Beer Recommender

## Summary

## EDA




## LDA Results

| Topic Number | Topics | Possible Category |
| --- | --- | --- |
| 0 |'ipa' 'double' 'imperial' 'black' 'kölsch' 'hop' | IPAs |
| 1 |'porter' 'altbier' 'imperial' 'russian' 'barrel' 'baltic'| Porters |
| 2 |'stout' 'oatmeal' 'winter' 'warmer' 'milk' 'sweet'| Stouts |
| 3 |'beer' 'pilsner' 'pilsener' 'rye' 'fruit' 'vegetable'| Pilsners |
| 4 |'ale' 'pale' 'lager' 'apa' 'amber' 'red'| APAs |
| 5 |'cider' 'berliner' 'apple' 'weissbier' 'hard' 'dry'| Ciders |
| 6 |'ale' 'blonde' 'witbier' 'strong' 'belgian' 'golden'|  Golden/Blondes |

## Input/Output Vectors

| Topic probabilities | ABU | IBU|
| --- | --- | --- |
|[0.01820796, 0.41002485, 0.01818473, 0.01818347, 0.01818821,0.01818524, 0.49902553]|1.21|0.62|
|... |... |... |

## Example

Recommendation of Beer within Dataset
```
{'Hard Apple': ['Ginger Cider', "Wolfman's Berliner", 'Schilling Hard Cider', 'Hard Cider', 'Nomader Weiss', 'Monkey Chased the Weasel', 'Magic Apple', 'Contemplation', 'Totally Radler']}
```

Recommendation for User's preference
```
{'target_beer': 'Voodoo Ranger Imperial IPA', 'recommendations': ['Valkyrie Double IPA', 'Northern Lights India Pale Ale', 'Northern Lights India Pale Ale', 'Jockamo IPA', 'White Reaper', 'Humidor Series India Pale Ale', 'Jai Alai IPA', 'Jai Alai IPA Aged on White Oak', 'The Great Return', 'Upslope Christmas Ale']}
```

Input

| Beer | ABV | IBU | Style |
| --- | --- | --- | --- |
| Voodoo Ranger Imperial IPA | 0.09 | 90 | Imperial IPA |

Recommendations

| Beer | ABV | IBU | Style |
| --- | --- | --- | --- |
| Valkyrie Double IPA | 0.092 | 100.0 | American IPA |
| Northern Lights India Pale Ale | 0.065 | 52.0| American IPA |
| Jockamo IPA | 0.065 | 52.0 | American IPA |
| White Reaper | 0.07 | 61.0 | Belgian IPA |
| Jai Alai IPA Aged on White Oak | 0.075 | 70.0 | American IPA |
| Humidor Series India Pale Ale | 0.075 | 70.0 | American IPA |
| The Great Return | 0.075 | 70.0 | American IPA|
| Jai Alai IPA | 0.075 | 70.0 | American IPA |
| Tripel Deke | 0.082 | 81.67 | Tripel |


## Demo
