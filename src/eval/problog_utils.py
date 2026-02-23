"""
Shared ProbLog helpers for evaluation.

Provides: SemiringDampened, quote_arg, build_score_lookups, get_score,
          apply_config_to_facts, rebuild_facts, execute_problog_direct.
"""

from problog.program import PrologString
from problog import get_evaluatable
from problog.evaluator import Semiring


# ─── Semirings ───────────────────────────────────────────────────────────────

class SemiringDampened(Semiring):
    """Dampened semiring: raises AND/OR combinations to alpha power.

    When alpha < 1, this prevents probabilities from shrinking too fast
    when many facts are chained together via AND.
    alpha=1.0 is equivalent to standard ProbLog.
    """
    def __init__(self, alpha=0.8):
        Semiring.__init__(self)
        self.alpha = alpha

    def one(self): return 1.0
    def zero(self): return 0.0
    def is_one(self, v): return v == 1.0
    def is_zero(self, v): return v == 0.0
    def plus(self, a, b): return 1.0 - (max(0, (1-a)*(1-b)))**self.alpha
    def times(self, a, b): return (max(0, a*b))**self.alpha
    def negate(self, a): return 1.0 - a
    def value(self, a): return float(a)
    def normalize(self, a, z): return a
    def result(self, a, formula=None): return float(a)
    def ad_negate(self, s, pos_only): return self.negate(s)
    def true(self, a=None): return 1.0
    def false(self, a=None): return 0.0
    def is_nsp(self): return True
    def pos_value(self, a, key=None): return float(a)
    def neg_value(self, a, key=None): return 1.0 - float(a)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def quote_arg(arg):
    """Quote a ProbLog argument for safe embedding in a program string."""
    if not isinstance(arg, str):
        return str(arg)
    if "'" in arg or " " in arg or "-" in arg or (arg and arg[0].isupper()):
        escaped = str(arg).replace("'", "\\'")
        return f"'{escaped}'"
    return arg


def build_score_lookups(sample):
    """Build lookup dicts from a sample's evidence for fast score retrieval."""
    attr_lookup = {}
    for attr in sample.get('evidence', {}).get('attributes', []):
        key = (attr.get('entity_id'), attr.get('value'))
        attr_lookup[key] = attr
    rel_lookup = {}
    for rel in sample.get('evidence', {}).get('relationships', []):
        key = (rel.get('subject_id'), rel.get('object_id'), rel.get('relation'))
        rel_lookup[key] = rel
    return attr_lookup, rel_lookup


def get_score(ev, score_type):
    """Get a score from an evidence dict, with support for averaged scores."""
    if score_type == 'avg_blip_qwen_tf':
        b = ev.get('blip_score')
        q = ev.get('qwen_tf_score')
        if b is not None and q is not None:
            return (b + q) / 2.0
        return b if b is not None else q
    else:
        return ev.get(score_type)


# ─── Fact building ───────────────────────────────────────────────────────────

def apply_config_to_facts(facts_data, attr_scores, rel_scores,
                          attr_score_type, rel_score_type,
                          entity_prob=None, agreement_mode=None):
    """Apply scoring config to a list of ProbLog fact dicts.

    Can be called during Phase 1 (with live EvidenceCollection score lists)
    or post-hoc (via rebuild_facts wrapper with stored JSON).

    Args:
        facts_data: list of {predicate, arguments, probability} dicts
        attr_scores: list of rich score dicts (with blip_score, qwen_tf_score, etc.)
        rel_scores: list of rich score dicts for relationships
        attr_score_type: which score to use for attributes (e.g. 'qwen_tf_score')
        rel_score_type: which score to use for relations (e.g. 'avg_blip_qwen_tf')
        entity_prob: optional override for entity fact probabilities
        agreement_mode: None, 'dampen_0.5', 'dampen_0.3', 'sharpen', or 'both'

    Returns: new list of fact dicts with updated probabilities
    """
    if not facts_data:
        return None

    # Build lookups from score lists
    attr_lookup = {}
    for attr in attr_scores:
        key = (attr.get('entity_id'), attr.get('value'))
        attr_lookup[key] = attr
    rel_lookup = {}
    for rel in rel_scores:
        key = (rel.get('subject_id'), rel.get('object_id'), rel.get('relation'))
        rel_lookup[key] = rel

    # Parse agreement mode into dampen/sharpen factors
    dampen_factor = None
    sharpen_factor = None
    if agreement_mode == 'dampen_0.5':
        dampen_factor = 0.5
    elif agreement_mode == 'dampen_0.3':
        dampen_factor = 0.3
    elif agreement_mode == 'sharpen':
        sharpen_factor = 1.5
    elif agreement_mode == 'both':
        dampen_factor = 0.5
        sharpen_factor = 1.5

    new_facts = []
    for fact in facts_data:
        pred = fact.get('predicate', '')
        args = fact.get('arguments', [])
        prob = fact.get('probability', 0.5)

        if pred == 'entity':
            if entity_prob is not None:
                prob = entity_prob

        elif pred == 'attribute' and len(args) >= 3:
            key = (args[1], args[2])
            ev = attr_lookup.get(key)
            if ev:
                raw = get_score(ev, attr_score_type)
                if raw is not None:
                    prob = raw
                if dampen_factor is not None or sharpen_factor is not None:
                    blip = ev.get('blip_score')
                    qwen = ev.get('qwen_tf_score')
                    if blip is not None and qwen is not None:
                        disagree = (blip >= 0.5) != (qwen >= 0.5)
                        if disagree and dampen_factor is not None:
                            prob = 0.5 + (prob - 0.5) * dampen_factor
                        elif not disagree and sharpen_factor is not None:
                            prob = 0.5 + (prob - 0.5) * sharpen_factor

        elif pred == 'relation' and len(args) >= 4:
            key = (args[1], args[2], args[3])
            ev = rel_lookup.get(key)
            if ev:
                raw = get_score(ev, rel_score_type)
                if raw is not None:
                    prob = raw
                if dampen_factor is not None or sharpen_factor is not None:
                    blip = ev.get('blip_score')
                    qwen = ev.get('qwen_tf_score')
                    if blip is not None and qwen is not None:
                        disagree = (blip >= 0.5) != (qwen >= 0.5)
                        if disagree and dampen_factor is not None:
                            prob = 0.5 + (prob - 0.5) * dampen_factor
                        elif not disagree and sharpen_factor is not None:
                            prob = 0.5 + (prob - 0.5) * sharpen_factor

        new_facts.append({**fact, 'probability': max(1e-7, min(1 - 1e-7, prob))})
    return new_facts


def rebuild_facts(sample, attr_score_type, rel_score_type, entity_prob=None, agreement_mode=None):
    """Rebuild ProbLog facts from a stored sample dict (backward-compat wrapper).

    Extracts facts and evidence from the sample dict and delegates to
    apply_config_to_facts().
    """
    stored_facts = sample.get('problog', {}).get('facts', [])
    if not stored_facts:
        return None
    attr_scores = sample.get('evidence', {}).get('attributes', [])
    rel_scores = sample.get('evidence', {}).get('relationships', [])
    return apply_config_to_facts(
        stored_facts, attr_scores, rel_scores,
        attr_score_type, rel_score_type,
        entity_prob=entity_prob, agreement_mode=agreement_mode)


# ─── Threshold ───────────────────────────────────────────────────────────────

def threshold_fn(n_facts, base, slope):
    """Log-based dynamic threshold: clip(base + slope * ln(n_facts + 1), 0, 1)."""
    import math
    return max(0.0, min(1.0, base + slope * math.log(n_facts + 1)))


# ─── ProbLog execution ───────────────────────────────────────────────────────

def execute_problog_direct(facts, rules, query, semiring=None):
    """Run ProbLog directly (no timeout). Used inside worker subprocesses."""
    lines = []
    for f in facts:
        args_str = ", ".join(quote_arg(a) for a in f['arguments'])
        p = max(1e-7, min(1 - 1e-7, f['probability']))
        lines.append(f"{p}::{f['predicate']}({args_str}).")
    program_str = "\n".join(lines) + f"\n\n{rules}\n\n{query}"
    try:
        if semiring is not None:
            kc_class = get_evaluatable(semiring=semiring)
            program = PrologString(program_str)
            kc = kc_class.create_from(program, semiring=semiring)
            result = kc.evaluate(semiring=semiring)
        else:
            program = PrologString(program_str)
            result = get_evaluatable().create_from(program).evaluate()
        for atom, prob in result.items():
            if 'answer' in str(atom):
                return float(prob)
        return 0.0
    except TimeoutError:
        raise
    except:
        return None
