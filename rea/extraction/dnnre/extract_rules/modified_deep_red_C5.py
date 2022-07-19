from rules.logic_manipulator.substitute_rules import substitute
from rules.rule import Rule
from rules.ruleset import Ruleset
from extraction.dnnre.extract_rules.dnnre_c5 import C5
from extraction.dnnre.model.trained_model import TrainedModel


# TODO change prints to debug logging

def extract_rules(model: TrainedModel, seed: int = 42, ):
    # Should be 1 DNF rule per class
    dnf_rules = set()

    for output_class in model.output_classes:
        output_layer = model.n_layers - 1

        # Total rule - Only keep 1 total rule in memory at a time
        total_rule = Rule.initial_rule(output_layer=output_layer,
                                       output_class=output_class,
                                       threshold=0.5)

        for hidden_layer in reversed(range(0, output_layer)):
            print('Extracting layer %d rules:' % hidden_layer)
            # Layerwise rules only store all rules for current layer
            im_rules = Ruleset()

            predictors = model.get_layer_activations(layer_index=hidden_layer)

            term_confidences = \
                total_rule.get_terms_with_conf_from_rule_premises()
            terms = term_confidences.keys()

            # how many terms iterating over
            for _ in terms:
                print('.', end='', flush=True)
            print()

            for term in terms:
                print('.', end='', flush=True)

                # y1', y2', ...ym' = t(h(x1)), t(h(x2)), ..., t(h(xm))
                target = term.apply(
                    model.get_layer_activations_of_neuron(
                        layer_index=hidden_layer + 1,
                        neuron_index=term.get_neuron_index()))

                prior_rule_conf = term_confidences[term]
                rule_conclusion_map = {True: term, False: term.negate()}
                im_rules.add_rules(C5(x=predictors, y=target,
                                      rule_conclusion_map=rule_conclusion_map,
                                      prior_rule_confidence=prior_rule_conf,
                                      seed=seed),
                                   )

            print('\nSubstituting layer %d rules' % hidden_layer, end=' ',
                  flush=True)
            total_rule = substitute(total_rule=total_rule,
                                    intermediate_rules=im_rules)
            print('done')

        dnf_rules.add(total_rule)

    return dnf_rules
