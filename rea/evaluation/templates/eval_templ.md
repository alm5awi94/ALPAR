
# Report for scenario: {{ dataset_kind }}

Rule Extraction Assistant report for {{ algo }} on {{ dataset_kind }} data.

- algorithm: `{{ algo }}`
- seed: `{{ seed }}`

## Extracted Rules

These are the rules extracted in a human-readable form:
```python
{% for rule in human_rules %}
{{ rule }}
{% endfor %}
```

And these are the rules with internal decoding:
```python
{% for rule in rules %}
{{ rule }}
{% endfor %}
```



### Rule Metrics

The following are metrics on the ruleset.

{% if algo == "alpa" %}
The loop of the ALPA algorithm you used found the following optimum:
- maximum fidelity achieved: `{{ max_fid }}`
- rho with maximum fidelity: `{{ best_rho }}`
{% endif %}

The following metrics are *per output class*:

|class:                  |{% for cls in output_classes: %}                "{{ cls }}"       |{% endfor %}
|-----------------------:|{% for cls in output_classes: %}:--------------------------------:|{% endfor %}
|n_clauses:              |{% for n_rules in n_clauses_per_class: %}       `{{ n_rules }}`   |{% endfor %}
|min_terms:              |{% for min_terms in min_n_terms: %}             `{{ min_terms }}` |{% endfor %}
|max_terms:              |{% for max_terms in max_n_terms: %}             `{{ max_terms }}` |{% endfor %}
|avg_terms:              |{% for avg_terms in av_n_terms_per_rule: %}     `{{ avg_terms }}` |{% endfor %}
|used features:          |{% for ft in feature_union_per_class: %}        `{{ ft }}`        |{% endfor %}
|features in every rule: |{% for ft in feature_intersection_per_class: %} `{{ ft }}`        |{% endfor %}

These metrics are over the *whole ruleset*:

- total number of rules: {{ n_rules }}
- common features (used by all classes): `{{ feature_intersec }}`
  - number of common features: `{{ feature_intersec_len }}`

## Accuracy and Fidelity

The accuracy and fidelity measures of the neural network compared to the rules.

- neural network:
  - accuracy: `{{ nn_acc|round(4) }}`
- rules:
  - accuracy: `{{ rules_acc|round(4) }}`
  - fidelity: `{{ rules_fid|round(4) }}`

## Confusion Matrices

![{{ title_prefix }} Rules Confusion Matrix](confusion_matrices/confusion_matrix_rules_on_{{ dataset_kind }}_data.png)

![{{ title_prefix }} NN Confusion Matrix](confusion_matrices/confusion_matrix_nn_on_{{ dataset_kind }}_data.png)

![{{ title_prefix }} NN to Rules Confusion Matrix](confusion_matrices/confusion_matrix_nn_vs_rules_on_{{ dataset_kind }}_data.png)

## Performance

- rule extraction:
    - time: `{{ time|round(4) }} sec`
    - memory: `{{ memory|round(4) }} MB`
