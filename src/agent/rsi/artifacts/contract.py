"""One output contract shared by planning, delivery, discovery and export."""
from agent.harness.contracts import digest

SCHEMA = 'scientist.artifact.v1'

WORKSPACE_INSTRUCTIONS = (
    'Keep one coherent project, not a new deployable model for every attempt. '
    'Use src/ for reusable implementation, training/ for reproduction, artifacts/ for selected fitted state, '
    'experiments/ for trial-specific code, metrics and checkpoints, reports/ for results, and data/ for inputs. '
    'Preserve uploaded paths; do not move or delete user inputs. Keep README.md current. '
    'Keep the same capability name across revisions of the same purpose and increment its version. '
    'Choose one final candidate per purpose using measured evidence; baselines, ablations and rejected '
    'generations remain experiments, not separate public tools. Different proteins, properties or scenarios '
    'may justify distinct capabilities. A reusable method follows the same packaging and independent Review '
    'as a model; fitted weights and predictive evaluation apply only to learned models. '
    'The host builds deliveries/ after admission; never write or edit host delivery files yourself.'
)


def guide():
    return {
        'schema': SCHEMA,
        'instructions': WORKSPACE_INSTRUCTIONS,
        'workspace': {
            'src/': 'Reusable implementation; preserve imports and relative paths.',
            'training/': 'Reproduction entry points and configuration.',
            'artifacts/': 'Selected fitted state, when the result is a learned model.',
            'experiments/': 'Baselines, iterations, ablations and intermediate checkpoints.',
            'reports/': 'Measured results, figures, method explanation and limitations.',
            'data/': 'Inputs and provenance; excluded from the registered runtime.',
            'deliveries/': 'Host-generated reviewed releases, never scratch output.',
        },
        'release': ['README.md', 'capability.json', 'MODEL_CARD.md or METHOD_CARD.md',
                    'predict.py', 'runtime/', 'interface.json', 'examples/', 'tests/',
                    'evaluation/', 'requirements.txt', 'checksums.json'],
        'selection': 'One selected release per research topic, kind and biological scope. Older releases retain their IDs and remain reproducible.',
    }


def family_id(goal_id, kind, scope):
    """Versions of one topic/purpose share a family; different topics stay separate."""
    scientific_scope = {key: sorted({str(x).strip().casefold() for x in scope.get(key, [])})
                        for key in ('proteins', 'properties', 'scenarios')}
    return digest({'topic': goal_id, 'kind': kind, 'scope': scientific_scope})


def selected_deliveries(deliveries):
    selected = {}
    for delivery in deliveries:
        selected[delivery.get('family_id') or delivery['candidate_id']] = delivery
    return list(selected.values())


def has_delivery(deliveries, kind):
    """Legacy deliveries were models; a method cannot satisfy a model goal."""
    return any(row.get('kind', 'model') == kind for row in selected_deliveries(deliveries))
