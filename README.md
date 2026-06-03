"""
╔══════════════════════════════════════════════════════════╗
║         EXPLAINABLE AI (XAI) REASONING ENGINE                              ║
║         Designed for B.Tech First Year Students                            ║
║         Version 1.0                                                        ║
╚══════════════════════════════════════════════════════════╝

What is XAI?
------------
Explainable AI is a set of processes and methods that allow humans to
understand and trust the results created by machine learning algorithms.
Instead of treating AI as a "black box", XAI makes the reasoning transparent.

This engine implements:
  1. Rule-Based Reasoning System
  2. Decision Tree with step-by-step explanation
  3. Feature Importance Calculator
  4. Confidence Score Tracker
  5. Human-Readable Explanation Generator
"""

import math
import random
from collections import defaultdict, Counter


# ============================================================
# SECTION 1: CORE DATA STRUCTURES
# ============================================================

class Fact:
    """
    A Fact represents a piece of known information.
    
    Think of it like a statement that is either True or False.
    Example: Fact("sky_is_blue", True) → "The sky is blue: TRUE"
    """
    def __init__(self, name: str, value, confidence: float = 1.0):
        """
        Parameters:
            name       : The name/label of this fact (e.g., "has_fever")
            value      : The value — can be True/False, a number, or a string
            confidence : How sure are we? 0.0 = not sure, 1.0 = totally sure
        """
        self.name = name
        self.value = value
        self.confidence = max(0.0, min(1.0, confidence))  # Clamp between 0 and 1

    def __repr__(self):
        return f"Fact({self.name}={self.value}, confidence={self.confidence:.2f})"


class Rule:
    """
    A Rule represents an IF-THEN logic statement.
    
    How it works:
        IF (condition1 AND condition2 AND ...) THEN conclusion
    
    Example (Medical):
        IF has_fever=True AND has_cough=True
        THEN diagnosis=flu (confidence=0.8)
    """
    def __init__(self, name: str, conditions: list, conclusion: Fact,
                 weight: float = 1.0, explanation: str = ""):
        """
        Parameters:
            name        : A short name for the rule (e.g., "flu_rule")
            conditions  : List of (fact_name, expected_value) tuples
            conclusion  : The Fact produced if all conditions match
            weight      : How important/reliable this rule is (0.0 to 1.0)
            explanation : A human-friendly description of WHY this rule exists
        """
        self.name = name
        self.conditions = conditions   # e.g., [("has_fever", True), ("has_cough", True)]
        self.conclusion = conclusion
        self.weight = weight
        self.explanation = explanation
        self.times_fired = 0           # Track how many times this rule was used

    def __repr__(self):
        cond_str = " AND ".join([f"{k}={v}" for k, v in self.conditions])
        return f"Rule({self.name}: IF {cond_str} THEN {self.conclusion.name}={self.conclusion.value})"


class InferenceTrace:
    """
    An InferenceTrace records every step the AI took to reach a conclusion.
    
    This is the KEY to explainability — without a trace, we can't explain
    HOW the AI reached its answer!
    """
    def __init__(self):
        self.steps = []          # Each step: (rule_used, facts_matched, conclusion_derived)
        self.facts_used = []     # All facts that were consulted
        self.rules_fired = []    # All rules that triggered

    def add_step(self, rule: Rule, matched_facts: list, derived: Fact):
        """Record one reasoning step."""
        self.steps.append({
            "rule": rule,
            "matched_facts": matched_facts,
            "derived": derived,
            "step_number": len(self.steps) + 1
        })
        self.rules_fired.append(rule)

    def summary(self) -> str:
        """Return a short summary of the reasoning process."""
        return (f"Reasoning used {len(self.rules_fired)} rule(s) "
                f"and checked {len(self.facts_used)} fact(s) "
                f"across {len(self.steps)} step(s).")


# ============================================================
# SECTION 2: THE KNOWLEDGE BASE
# ============================================================

class KnowledgeBase:
    """
    The KnowledgeBase stores all Facts and Rules that the AI knows.
    
    Think of it as the AI's "memory" and "logic book" combined.
    
    Structure:
        facts  → What we currently know to be true
        rules  → The logic patterns to apply to those facts
    """
    def __init__(self, name: str = "KnowledgeBase"):
        self.name = name
        self.facts: dict = {}    # fact_name → Fact object
        self.rules: list = []    # List of Rule objects
        self.history = []        # Log of all changes

    def add_fact(self, fact: Fact):
        """Add or update a fact in the knowledge base."""
        old = self.facts.get(fact.name)
        self.facts[fact.name] = fact
        self.history.append(f"[FACT ADDED] {fact}")
        if old:
            self.history.append(f"  └─ Updated from: {old}")
        return self

    def add_rule(self, rule: Rule):
        """Add a reasoning rule to the knowledge base."""
        self.rules.append(rule)
        self.history.append(f"[RULE ADDED] {rule.name}")
        return self

    def get_fact(self, name: str):
        """Retrieve a fact by name (returns None if not found)."""
        return self.facts.get(name)

    def remove_fact(self, name: str):
        """Remove a fact from the knowledge base."""
        if name in self.facts:
            removed = self.facts.pop(name)
            self.history.append(f"[FACT REMOVED] {removed}")
            return True
        return False

    def display(self):
        """Pretty-print the current state of the knowledge base."""
        print(f"\n{'='*60}")
        print(f"  KNOWLEDGE BASE: {self.name}")
        print(f"{'='*60}")
        print(f"\n  📌 FACTS ({len(self.facts)}):")
        if not self.facts:
            print("     (empty)")
        for name, fact in self.facts.items():
            bar = "█" * int(fact.confidence * 10) + "░" * (10 - int(fact.confidence * 10))
            print(f"     • {name:<25} = {str(fact.value):<10} confidence: [{bar}] {fact.confidence:.0%}")

        print(f"\n  📋 RULES ({len(self.rules)}):")
        if not self.rules:
            print("     (empty)")
        for rule in self.rules:
            print(f"     • {rule.name:<20} → {rule.conclusion.name} = {rule.conclusion.value}")
            if rule.explanation:
                print(f"       Reason: {rule.explanation}")
        print(f"{'='*60}\n")


# ============================================================
# SECTION 3: THE INFERENCE ENGINE (The AI Brain)
# ============================================================

class InferenceEngine:
    """
    The InferenceEngine is the "thinking" part of the AI.
    
    It uses FORWARD CHAINING: Starting from known facts, it applies
    rules to derive NEW facts, and keeps going until no more rules fire.
    
    Step-by-Step Forward Chaining:
    ┌─────────────┐    ┌──────────────┐    ┌────────────────┐
    │  Known Facts │ →  │  Match Rules  │ →  │ Derive New Facts│
    └─────────────┘    └──────────────┘    └────────────────┘
                              ↑___________________________|
                              (repeat until no new facts found)
    """
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.trace = InferenceTrace()
        self.conflict_resolution = "priority"  # How to handle multiple matching rules

    def check_condition(self, fact_name: str, expected_value) -> tuple:
        """
        Check if a single condition is satisfied.
        
        Returns: (is_satisfied, actual_fact_or_None)
        """
        fact = self.kb.get_fact(fact_name)
        if fact is None:
            return False, None

        # Support comparison operators in expected values
        if isinstance(expected_value, str) and expected_value.startswith(">"):
            threshold = float(expected_value[1:])
            return (isinstance(fact.value, (int, float)) and fact.value > threshold), fact
        elif isinstance(expected_value, str) and expected_value.startswith("<"):
            threshold = float(expected_value[1:])
            return (isinstance(fact.value, (int, float)) and fact.value < threshold), fact
        elif isinstance(expected_value, str) and expected_value.startswith(">="):
            threshold = float(expected_value[2:])
            return (isinstance(fact.value, (int, float)) and fact.value >= threshold), fact
        else:
            return fact.value == expected_value, fact

    def evaluate_rule(self, rule: Rule) -> tuple:
        """
        Check if ALL conditions in a rule are satisfied.
        
        Returns: (all_satisfied, list_of_matched_facts, combined_confidence)
        """
        matched_facts = []
        combined_confidence = 1.0

        for (fact_name, expected_value) in rule.conditions:
            satisfied, fact = self.check_condition(fact_name, expected_value)
            if not satisfied:
                return False, [], 0.0
            matched_facts.append(fact)
            # Confidence chains multiply (weakest link principle)
            combined_confidence *= fact.confidence

        return True, matched_facts, combined_confidence

    def forward_chain(self, max_iterations: int = 100) -> list:
        """
        Perform Forward Chaining Inference.
        
        This is the main reasoning loop:
        1. Check all rules against current facts
        2. Fire rules whose conditions are met
        3. Add newly derived facts to the knowledge base
        4. Repeat until nothing new can be derived
        
        Returns: List of newly derived facts
        """
        self.trace = InferenceTrace()
        derived_facts = []
        iteration = 0

        print(f"\n{'─'*60}")
        print("  🧠 INFERENCE ENGINE: FORWARD CHAINING STARTED")
        print(f"{'─'*60}")

        while iteration < max_iterations:
            iteration += 1
            new_derivations_this_round = []

            for rule in self.kb.rules:
                # Skip if conclusion already known (derived in a previous iteration)
                existing = self.kb.get_fact(rule.conclusion.name)
                if existing and existing.value == rule.conclusion.value and rule.name in [s["rule"].name for s in self.trace.steps]:
                    continue

                # Evaluate the rule
                satisfied, matched_facts, conf = self.evaluate_rule(rule)

                if satisfied:
                    # Calculate final confidence for the new fact
                    final_confidence = conf * rule.weight * rule.conclusion.confidence
                    new_fact = Fact(
                        rule.conclusion.name,
                        rule.conclusion.value,
                        final_confidence
                    )

                    print(f"\n  ✅ ITERATION {iteration} | Rule Fired: [{rule.name}]")
                    print(f"     Conditions matched:")
                    for f in matched_facts:
                        print(f"       • {f.name} = {f.value} (confidence: {f.confidence:.0%})")
                    print(f"     ⟹ Derived: {new_fact.name} = {new_fact.value}")
                    print(f"     ⟹ Confidence: {final_confidence:.0%}")
                    if rule.explanation:
                        print(f"     💡 Reason: {rule.explanation}")

                    self.kb.add_fact(new_fact)
                    self.trace.add_step(rule, matched_facts, new_fact)
                    new_derivations_this_round.append(new_fact)
                    rule.times_fired += 1
                    derived_facts.append(new_fact)

            if not new_derivations_this_round:
                print(f"\n  🏁 No new facts derived in iteration {iteration}. Stopping.")
                break

        print(f"\n{'─'*60}")
        print(f"  Inference complete. Derived {len(derived_facts)} new fact(s).")
        print(f"{'─'*60}\n")
        return derived_facts

    def get_explanation(self, fact_name: str) -> str:
        """
        Generate a human-readable explanation for HOW a conclusion was reached.
        
        This is the core of Explainability — turning machine logic into
        plain language that a human can understand and verify.
        """
        lines = []
        lines.append(f"\n{'═'*60}")
        lines.append(f"  📖 EXPLANATION: How did the AI decide '{fact_name}'?")
        lines.append(f"{'═'*60}")

        # Find all steps that contributed to this fact
        relevant_steps = [s for s in self.trace.steps if s["derived"].name == fact_name]

        if not relevant_steps:
            fact = self.kb.get_fact(fact_name)
            if fact:
                lines.append(f"\n  This fact ({fact_name} = {fact.value}) was given as input,")
                lines.append(f"  not derived. It's a base fact with {fact.confidence:.0%} confidence.")
            else:
                lines.append(f"\n  ❌ No explanation found. '{fact_name}' was not derived or known.")
            return "\n".join(lines)

        for step in relevant_steps:
            rule = step["rule"]
            lines.append(f"\n  STEP {step['step_number']}: Applied Rule '{rule.name}'")
            lines.append(f"  {'─'*50}")
            lines.append(f"  BECAUSE the AI observed:")
            for f in step["matched_facts"]:
                lines.append(f"    • {f.name} is {f.value} (I am {f.confidence:.0%} sure)")
            lines.append(f"\n  THEREFORE it concluded:")
            lines.append(f"    → {step['derived'].name} = {step['derived'].value}")
            lines.append(f"    → Confidence: {step['derived'].confidence:.0%}")
            if rule.explanation:
                lines.append(f"\n  WHY this rule exists:")
                lines.append(f"    \"{rule.explanation}\"")

        lines.append(f"\n{'═'*60}\n")
        return "\n".join(lines)


# ============================================================
# SECTION 4: DECISION TREE (Another Explainable AI Method)
# ============================================================

class DecisionNode:
    """
    A single node in a Decision Tree.
    
    A Decision Tree is like a flowchart:
    - Internal nodes ask a YES/NO question
    - Leaf nodes give the final answer
    
    Example:
              [Temperature > 38?]
               /              \\
            YES               NO
          [Cough?]         [No fever]
          /    \\
        YES     NO
      [Flu]  [Allergies]
    """
    def __init__(self, question: str = None, feature: str = None,
                 threshold=None, yes_node=None, no_node=None,
                 answer: str = None, confidence: float = 1.0):
        self.question = question    # Human-readable question
        self.feature = feature      # Which fact/feature to check
        self.threshold = threshold  # Value to compare against
        self.yes_node = yes_node    # Node if condition is True
        self.no_node = no_node      # Node if condition is False
        self.answer = answer        # Only set for leaf nodes
        self.confidence = confidence


class DecisionTree:
    """
    A transparent Decision Tree Classifier with path explanation.
    
    Unlike neural networks, Decision Trees are naturally explainable
    because you can trace EXACTLY which questions led to each answer.
    """
    def __init__(self, name: str):
        self.name = name
        self.root = None
        self.decision_path = []  # Track the path taken for last prediction

    def predict(self, facts: dict) -> tuple:
        """
        Traverse the tree using given facts.
        
        Returns: (answer, confidence, decision_path)
        """
        self.decision_path = []
        node = self.root
        depth = 0

        print(f"\n  🌳 DECISION TREE: {self.name}")
        print(f"  {'─'*50}")

        while node is not None:
            if node.answer is not None:
                # Leaf node — we have our answer
                print(f"  {'  '*depth}└─ ANSWER: {node.answer} (confidence: {node.confidence:.0%})")
                return node.answer, node.confidence, self.decision_path

            # Internal node — ask the question
            feature_value = facts.get(node.feature)
            if feature_value is None:
                print(f"  {'  '*depth}⚠️  Missing feature: {node.feature}. Cannot proceed.")
                return "Unknown", 0.0, self.decision_path

            # Evaluate the condition
            if isinstance(node.threshold, bool):
                condition_met = (feature_value == node.threshold)
            elif isinstance(node.threshold, (int, float)):
                condition_met = (feature_value > node.threshold)
            else:
                condition_met = (feature_value == node.threshold)

            arrow = "✓ YES" if condition_met else "✗ NO"
            print(f"  {'  '*depth}❓ {node.question}")
            print(f"  {'  '*depth}   → {feature_value} → {arrow}")

            self.decision_path.append({
                "question": node.question,
                "feature": node.feature,
                "value": feature_value,
                "answer": condition_met
            })

            node = node.yes_node if condition_met else node.no_node
            depth += 1

        return "Unknown", 0.0, self.decision_path

    def explain_path(self) -> str:
        """Generate a narrative explanation of the decision path."""
        if not self.decision_path:
            return "No decision has been made yet."

        lines = ["\n  📝 DECISION PATH EXPLANATION:"]
        for i, step in enumerate(self.decision_path, 1):
            verdict = "YES" if step["answer"] else "NO"
            lines.append(f"  Step {i}: Asked '{step['question']}'")
            lines.append(f"          Value was '{step['value']}' → Answer: {verdict}")
        return "\n".join(lines)


# ============================================================
# SECTION 5: FEATURE IMPORTANCE (Why did THESE inputs matter?)
# ============================================================

class FeatureImportanceAnalyzer:
    """
    Calculates and explains WHICH input features mattered most for a decision.
    
    This answers the question: "What were the most important clues?"
    
    Method used: Rule Coverage Analysis
    - A feature is more important if it appears in more rules that fired
    - A feature is more important if those rules had higher weights/confidence
    """
    def __init__(self):
        self.importance_scores = {}

    def analyze(self, trace: InferenceTrace, all_facts: dict) -> dict:
        """
        Analyze the trace to compute feature importance scores.
        
        Returns: dict of {feature_name: importance_score}
        """
        scores = defaultdict(float)
        total_weight = 0.0

        for step in trace.steps:
            rule = step["rule"]
            rule_importance = rule.weight * step["derived"].confidence

            for fact in step["matched_facts"]:
                scores[fact.name] += rule_importance
                total_weight += rule_importance

        # Normalize scores to 0–1 range
        if total_weight > 0:
            for key in scores:
                scores[key] /= total_weight

        self.importance_scores = dict(scores)
        return self.importance_scores

    def display(self):
        """Print a bar chart of feature importances."""
        if not self.importance_scores:
            print("  No importance scores computed yet.")
            return

        sorted_items = sorted(self.importance_scores.items(), key=lambda x: x[1], reverse=True)

        print(f"\n{'─'*60}")
        print("  📊 FEATURE IMPORTANCE ANALYSIS")
        print("  (Which inputs influenced the AI's decision most?)")
        print(f"{'─'*60}")

        for feature, score in sorted_items:
            bar_len = int(score * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            print(f"  {feature:<25} [{bar}] {score:.1%}")
        print(f"{'─'*60}\n")


# ============================================================
# SECTION 6: CONFIDENCE TRACKER
# ============================================================

class ConfidenceTracker:
    """
    Tracks and visualizes confidence scores throughout reasoning.
    
    Confidence tells us: "How SURE is the AI about each conclusion?"
    
    Scale:
        0.0 – 0.3  : Low confidence (guessing)
        0.3 – 0.6  : Moderate confidence (reasonable guess)
        0.6 – 0.8  : High confidence (well-supported)
        0.8 – 1.0  : Very high confidence (near certain)
    """
    def __init__(self):
        self.records = []

    def record(self, label: str, confidence: float):
        """Record a confidence score with a label."""
        self.records.append((label, confidence))

    def from_trace(self, trace: InferenceTrace):
        """Auto-populate from an inference trace."""
        for step in trace.steps:
            self.record(step["derived"].name, step["derived"].confidence)

    def display(self):
        """Print a confidence dashboard."""
        if not self.records:
            print("  No confidence records yet.")
            return

        print(f"\n{'─'*60}")
        print("  🎯 CONFIDENCE DASHBOARD")
        print(f"{'─'*60}")

        for label, confidence in self.records:
            if confidence >= 0.8:
                emoji = "🟢"
                rating = "HIGH"
            elif confidence >= 0.5:
                emoji = "🟡"
                rating = "MEDIUM"
            else:
                emoji = "🔴"
                rating = "LOW"

            bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
            print(f"  {emoji} {label:<25} [{bar}] {confidence:.0%}  ({rating})")

        avg = sum(c for _, c in self.records) / len(self.records)
        print(f"{'─'*60}")
        print(f"  Average confidence: {avg:.0%}")
        print(f"{'─'*60}\n")


# ============================================================
# SECTION 7: COMPLETE XAI SYSTEM (Putting it all together)
# ============================================================

class XAIReasoningEngine:
    """
    The complete Explainable AI Reasoning Engine.
    
    Combines:
    ┌─────────────────────────────────────────────────────┐
    │  Knowledge Base   → Stores facts & rules            │
    │  Inference Engine → Forward chaining reasoning      │
    │  Decision Tree    → Transparent classification      │
    │  Feature Analyzer → Importance scores               │
    │  Confidence Track → Certainty monitoring            │
    │  Explanation Gen  → Human-readable outputs          │
    └─────────────────────────────────────────────────────┘
    """
    def __init__(self, name: str):
        self.name = name
        self.kb = KnowledgeBase(name)
        self.engine = InferenceEngine(self.kb)
        self.feature_analyzer = FeatureImportanceAnalyzer()
        self.confidence_tracker = ConfidenceTracker()
        self.decision_tree = None

    def load_facts(self, fact_dict: dict):
        """Load multiple facts at once from a dictionary."""
        print(f"\n  📥 Loading {len(fact_dict)} facts into knowledge base...")
        for name, (value, confidence) in fact_dict.items():
            self.kb.add_fact(Fact(name, value, confidence))
            print(f"     ✓ {name} = {value} (confidence: {confidence:.0%})")

    def add_rule(self, name, conditions, conclusion_name, conclusion_value,
                 conclusion_confidence, weight=1.0, explanation=""):
        """Convenience method to add a rule."""
        conclusion = Fact(conclusion_name, conclusion_value, conclusion_confidence)
        rule = Rule(name, conditions, conclusion, weight, explanation)
        self.kb.add_rule(rule)

    def reason(self) -> list:
        """Run the full reasoning process and return derived facts."""
        print(f"\n{'╔'+'═'*58+'╗'}")
        print(f"{'║':<3} XAI ENGINE: {self.name:<46}{'║'}")
        print(f"{'╚'+'═'*58+'╝'}")

        # Step 1: Show initial state
        self.kb.display()

        # Step 2: Run inference
        derived = self.engine.forward_chain()

        # Step 3: Analyze feature importance
        self.feature_analyzer.analyze(self.engine.trace, self.kb.facts)

        # Step 4: Track confidence
        self.confidence_tracker.from_trace(self.engine.trace)

        return derived

    def explain(self, fact_name: str):
        """Print full explanation for a specific conclusion."""
        print(self.engine.get_explanation(fact_name))

    def full_report(self):
        """Print complete analysis report."""
        print(f"\n{'╔'+'═'*58+'╗'}")
        print(f"{'║':<3} FULL EXPLAINABILITY REPORT: {self.name:<30}{'║'}")
        print(f"{'╚'+'═'*58+'╝'}")
        self.feature_analyzer.display()
        self.confidence_tracker.display()
        print(f"\n  📋 REASONING SUMMARY:")
        print(f"  {self.engine.trace.summary()}")

        if self.engine.trace.rules_fired:
            print(f"\n  Rules that fired:")
            rule_counts = Counter(r.name for r in self.engine.trace.rules_fired)
            for rule_name, count in rule_counts.items():
                print(f"    • {rule_name} (fired {count}x)")


# ============================================================
# SECTION 8: DEMO SCENARIOS
# ============================================================

def demo_medical_diagnosis():
    """
    Demo 1: Medical Symptom Checker
    
    Scenario: A patient comes in with symptoms.
    The AI reasons through the symptoms to suggest a possible diagnosis.
    
    ⚠️ Disclaimer: This is for educational purposes only.
                   Always consult a real doctor!
    """
    print("\n" + "="*60)
    print("  DEMO 1: MEDICAL SYMPTOM CHECKER")
    print("="*60)

    engine = XAIReasoningEngine("MedicalDiagnosis")

    # Load patient's symptoms as facts
    engine.load_facts({
        "has_fever":        (True,  0.95),   # Patient definitely has fever
        "temperature_c":    (38.8,  0.99),   # Measured temperature
        "has_cough":        (True,  0.90),   # Patient reports cough
        "has_body_aches":   (True,  0.85),   # Patient reports body aches
        "has_runny_nose":   (False, 0.80),   # No runny nose
        "has_sore_throat":  (False, 0.75),   # No sore throat
        "exposure_to_sick": (True,  0.70),   # Was around sick people
    })

    # Define medical reasoning rules
    engine.add_rule(
        name="high_fever_rule",
        conditions=[("temperature_c", ">38.0")],
        conclusion_name="high_fever_confirmed",
        conclusion_value=True,
        conclusion_confidence=0.95,
        weight=0.90,
        explanation="A temperature above 38°C is medically defined as high fever"
    )

    engine.add_rule(
        name="flu_rule",
        conditions=[
            ("high_fever_confirmed", True),
            ("has_cough", True),
            ("has_body_aches", True),
            ("exposure_to_sick", True),
        ],
        conclusion_name="likely_diagnosis",
        conclusion_value="Influenza (Flu)",
        conclusion_confidence=0.85,
        weight=0.88,
        explanation="Combination of high fever, cough, body aches and exposure "
                    "is the classic presentation of influenza"
    )

    engine.add_rule(
        name="rest_and_fluids_rule",
        conditions=[("likely_diagnosis", "Influenza (Flu)")],
        conclusion_name="recommended_action",
        conclusion_value="Rest, fluids, consult doctor for antivirals",
        conclusion_confidence=0.90,
        weight=0.95,
        explanation="Standard first-line treatment for flu includes rest and hydration"
    )

    # Run reasoning
    engine.reason()

    # Explain specific conclusions
    engine.explain("likely_diagnosis")
    engine.explain("recommended_action")

    # Full report
    engine.full_report()


def demo_loan_approval():
    """
    Demo 2: Bank Loan Approval System
    
    Scenario: A bank AI decides whether to approve a loan.
    We make the decision process completely transparent.
    """
    print("\n" + "="*60)
    print("  DEMO 2: LOAN APPROVAL DECISION SYSTEM")
    print("="*60)

    engine = XAIReasoningEngine("LoanApproval")

    # Applicant information
    engine.load_facts({
        "credit_score":      (720,   0.99),  # High credit score
        "annual_income_usd": (65000, 0.99),  # Annual income
        "loan_amount_usd":   (20000, 0.99),  # Requested loan
        "debt_to_income":    (0.28,  0.95),  # 28% of income goes to debt
        "employment_years":  (5,     0.90),  # 5 years employed
        "has_collateral":    (True,  0.85),  # Has collateral
        "previous_defaults": (0,     0.99),  # No previous defaults
    })

    # Loan eligibility rules
    engine.add_rule(
        name="good_credit_rule",
        conditions=[("credit_score", ">699")],
        conclusion_name="credit_approved",
        conclusion_value=True,
        conclusion_confidence=0.92,
        weight=0.95,
        explanation="Credit score above 700 indicates reliable repayment history"
    )

    engine.add_rule(
        name="stable_income_rule",
        conditions=[
            ("annual_income_usd", ">50000"),
            ("employment_years", ">2"),
        ],
        conclusion_name="income_stable",
        conclusion_value=True,
        conclusion_confidence=0.88,
        weight=0.90,
        explanation="Income >$50K with 2+ years employment shows financial stability"
    )

    engine.add_rule(
        name="low_risk_rule",
        conditions=[
            ("debt_to_income", "<0.35"),
            ("previous_defaults", False)
        ],
        conclusion_name="low_default_risk",
        conclusion_value=True,
        conclusion_confidence=0.85,
        weight=0.88,
        explanation="DTI below 35% with no defaults indicates low repayment risk"
    )

    engine.add_rule(
        name="loan_approved_rule",
        conditions=[
            ("credit_approved", True),
            ("income_stable", True),
            ("low_default_risk", True),
        ],
        conclusion_name="loan_decision",
        conclusion_value="APPROVED",
        conclusion_confidence=0.90,
        weight=0.95,
        explanation="All three pillars (credit, income, risk) are satisfactory"
    )

    # Run reasoning
    engine.reason()

    # Full explanation of loan decision
    engine.explain("loan_decision")

    # Full report
    engine.full_report()


def demo_decision_tree_spam():
    """
    Demo 3: Email Spam Classifier using Decision Tree
    
    Shows how a Decision Tree provides a transparent,
    step-by-step classification with full path explanation.
    """
    print("\n" + "="*60)
    print("  DEMO 3: SPAM EMAIL DETECTOR (Decision Tree)")
    print("="*60)

    # Build the decision tree manually (in real ML, this is learned from data)
    #
    #            [Has suspicious words?]
    #             /                   \\
    #           YES                    NO
    #     [From unknown?]        [Has links?]
    #       /        \\              /      \\
    #     YES         NO          YES       NO
    #   [SPAM]    [Check       [Check]    [NOT
    #             subject]               SPAM]
    #

    tree = DecisionTree("SpamClassifier")

    # Build tree nodes (from leaves up to root)
    spam_leaf = DecisionNode(answer="🚫 SPAM", confidence=0.94)
    notspam_leaf_1 = DecisionNode(answer="✅ NOT SPAM", confidence=0.88)
    notspam_leaf_2 = DecisionNode(answer="✅ NOT SPAM", confidence=0.91)
    maybe_leaf = DecisionNode(answer="⚠️ REVIEW MANUALLY", confidence=0.60)

    # Second-level nodes
    unknown_sender_node = DecisionNode(
        question="Is sender unknown / not in contacts?",
        feature="unknown_sender",
        threshold=True,
        yes_node=spam_leaf,
        no_node=notspam_leaf_1
    )

    links_node = DecisionNode(
        question="Does email contain suspicious links?",
        feature="has_suspicious_links",
        threshold=True,
        yes_node=maybe_leaf,
        no_node=notspam_leaf_2
    )

    # Root node
    tree.root = DecisionNode(
        question="Does email contain suspicious/spam words?",
        feature="has_spam_words",
        threshold=True,
        yes_node=unknown_sender_node,
        no_node=links_node
    )

    # Test email 1: Obvious spam
    print("\n  📧 EMAIL 1: 'CONGRATULATIONS! You won $1,000,000!'")
    email_1 = {
        "has_spam_words": True,
        "unknown_sender": True,
        "has_suspicious_links": True,
    }
    answer, confidence, path = tree.predict(email_1)
    print(f"\n  Classification: {answer}")
    print(f"  Confidence: {confidence:.0%}")
    print(tree.explain_path())

    # Test email 2: Legitimate email
    print("\n  📧 EMAIL 2: 'Meeting rescheduled to 3 PM tomorrow'")
    email_2 = {
        "has_spam_words": False,
        "unknown_sender": False,
        "has_suspicious_links": False,
    }
    answer, confidence, path = tree.predict(email_2)
    print(f"\n  Classification: {answer}")
    print(f"  Confidence: {confidence:.0%}")
    print(tree.explain_path())


def demo_student_grade_prediction():
    """
    Demo 4: Student Grade Prediction
    
    A relatable example for B.Tech students!
    Predicts whether a student will pass/fail based on study habits.
    """
    print("\n" + "="*60)
    print("  DEMO 4: STUDENT PERFORMANCE PREDICTOR")
    print("="*60)

    engine = XAIReasoningEngine("StudentPerformance")

    # Student profile
    engine.load_facts({
        "hours_studied_per_day":  (4.5,  0.90),  # Hours studied daily
        "attendance_percent":     (85,   0.95),  # Class attendance
        "completed_assignments":  (True, 0.99),  # Did all assignments
        "practice_problems_done": (True, 0.85),  # Solved extra problems
        "gets_enough_sleep":      (True, 0.80),  # Healthy sleep
        "distraction_hours":      (1.5,  0.75),  # Hours on social media/games
    })

    # Academic success rules
    engine.add_rule(
        name="dedicated_study_rule",
        conditions=[
            ("hours_studied_per_day", ">3.0"),
            ("completed_assignments", True),
        ],
        conclusion_name="study_habits_good",
        conclusion_value=True,
        conclusion_confidence=0.90,
        weight=0.92,
        explanation="Consistent daily study + assignment completion is the "
                    "strongest predictor of academic performance"
    )

    engine.add_rule(
        name="attendance_rule",
        conditions=[("attendance_percent", ">75")],
        conclusion_name="attendance_sufficient",
        conclusion_value=True,
        conclusion_confidence=0.95,
        weight=0.88,
        explanation="Above 75% attendance ensures all key topics are covered"
    )

    engine.add_rule(
        name="holistic_performance_rule",
        conditions=[
            ("study_habits_good", True),
            ("attendance_sufficient", True),
            ("practice_problems_done", True),
        ],
        conclusion_name="expected_result",
        conclusion_value="PASS with Distinction",
        conclusion_confidence=0.87,
        weight=0.90,
        explanation="Strong study habits + attendance + practice = high performance"
    )

    # Run reasoning
    engine.reason()
    engine.explain("expected_result")
    engine.full_report()


# ============================================================
# SECTION 9: MAIN ENTRY POINT
# ============================================================

def main():
    """
    Run all demos to showcase the XAI Reasoning Engine.
    """
    print("\n")
    print("█" * 62)
    print("█" + " " * 60 + "█")
    print("█   EXPLAINABLE AI (XAI) REASONING ENGINE                  █")
    print("█   B.Tech First Year — Educational Demo                   █")
    print("█" + " " * 60 + "█")
    print("█" * 62)
    print("""
  This engine demonstrates core concepts of Explainable AI:
  
  1. 🧠 Rule-Based Inference (Forward Chaining)
  2. 🌳 Decision Trees (Transparent Classification)  
  3. 📊 Feature Importance (What mattered most?)
  4. 🎯 Confidence Scores (How sure is the AI?)
  5. 📖 Human-Readable Explanations (WHY did AI decide this?)
""")

    # Run all four demos
    demo_medical_diagnosis()
    demo_loan_approval()
    demo_decision_tree_spam()
    demo_student_grade_prediction()

    print("\n" + "="*62)
    print("  ✅ All demos complete!")
    print("  See documentation.md for full learning guide.")
    print("="*62 + "\n")
if __name__ == "__main__":
    main()
