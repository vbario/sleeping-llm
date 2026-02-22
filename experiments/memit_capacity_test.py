"""MEMIT capacity test — finds the ceiling for simultaneous MEMIT edits.

Injects increasing numbers of facts and measures recall at each level
to find where MEMIT degrades. This determines nap/sleep frequency.

Usage:
    python experiments/memit_capacity_test.py --config experiments/configs/3b_memit.yaml
    python experiments/memit_capacity_test.py --config experiments/configs/70b_memit.yaml --max-facts 100
    python experiments/memit_capacity_test.py --config experiments/configs/8b_memit.yaml --batch-size 10
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.orchestrator import Orchestrator
from src.memory.memit import FactTriple


# --- Synthetic fact generator ---

FIRST_NAMES = [
    "Elena", "Marcus", "Priya", "Tobias", "Yuki", "Carlos", "Fatima", "Henrik",
    "Amara", "Diego", "Leila", "Oscar", "Nia", "Felix", "Sana", "Ivan",
    "Zara", "Mateo", "Aisha", "Liam", "Rosa", "Kai", "Mira", "Ezra",
    "Luna", "Theo", "Noor", "Axel", "Vera", "Hugo", "Ines", "Ravi",
    "Cleo", "Emil", "Dina", "Omar", "Iris", "Leo", "Maya", "Nils",
    "Aria", "Jude", "Tara", "Sven", "Alma", "Rex", "Faye", "Amos",
    "Greta", "Idris", "Lily", "Bo", "Uma", "Troy", "Wren", "Quinn",
    "Sage", "Rio", "Pearl", "Ace", "Nova", "Cruz", "Ivy", "Dale",
    "Opal", "Finn", "June", "Bram", "Elle", "Kit", "Thea", "Zeke",
    "Neva", "Cole", "Hope", "Drew", "Esme", "Gage", "Lena", "Nash",
]

LAST_NAMES = [
    "Voronov", "Takahashi", "Lindström", "Okafor", "Petrov", "Navarro", "Kim",
    "Andersen", "Morales", "Fujimoto", "Bergström", "Osei", "Volkov", "Torres",
    "Tanaka", "Johansson", "Adeyemi", "Kozlov", "Santos", "Watanabe",
    "Eriksson", "Mensah", "Sokolov", "Reyes", "Nakamura", "Nilsson", "Boateng",
    "Fedorov", "Cruz", "Yamamoto", "Larsson", "Asante", "Popov", "Ruiz",
    "Ito", "Olsson", "Owusu", "Kuznetsov", "Flores", "Sato",
]

CITIES = [
    "Portland", "Austin", "Denver", "Seattle", "Boston", "Nashville", "Minneapolis",
    "Savannah", "Tucson", "Madison", "Asheville", "Boise", "Burlington", "Charleston",
    "Duluth", "Eugene", "Flagstaff", "Galveston", "Helena", "Ithaca",
    "Jacksonville", "Knoxville", "Louisville", "Memphis", "Norfolk",
    "Olympia", "Pittsburgh", "Richmond", "Salem", "Topeka",
    "Urbana", "Valdosta", "Wichita", "Yuma", "Zanesville",
    "Anchorage", "Bismarck", "Columbus", "Detroit", "El Paso",
]

JOBS = [
    "software engineer", "teacher", "nurse", "architect", "chef",
    "photographer", "journalist", "dentist", "librarian", "mechanic",
    "pilot", "veterinarian", "electrician", "botanist", "translator",
    "paramedic", "sculptor", "auditor", "geologist", "florist",
    "plumber", "cartographer", "pharmacist", "welder", "curator",
    "firefighter", "optometrist", "beekeeper", "locksmith", "tailor",
    "surveyor", "brewer", "midwife", "ranger", "sommelier",
    "typographer", "glazier", "luthier", "falconer", "farrier",
]

PETS = [
    ("dog", ["golden retriever", "labrador", "beagle", "poodle", "bulldog",
             "dachshund", "husky", "corgi", "dalmatian", "boxer"]),
    ("cat", ["siamese", "persian", "maine coon", "tabby", "bengal",
             "ragdoll", "sphynx", "abyssinian", "british shorthair", "burmese"]),
]

PET_NAMES = [
    "Biscuit", "Maple", "Ziggy", "Pepper", "Mochi", "Hazel", "Cosmo", "Olive",
    "Finn", "Clover", "Jasper", "Poppy", "Bruno", "Noodle", "Scout", "Willow",
    "Gizmo", "Cinnamon", "Rusty", "Peaches", "Bandit", "Rosie", "Thor", "Daisy",
    "Loki", "Honey", "Rex", "Sage", "Duke", "Luna", "Bear", "Stella",
    "Rocky", "Penny", "Max", "Ruby", "Zeus", "Coco", "Ace", "Bella",
]

COLORS = [
    "blue", "green", "red", "purple", "orange", "yellow", "teal", "maroon",
    "indigo", "coral", "emerald", "crimson", "gold", "silver", "navy",
    "turquoise", "magenta", "amber", "lavender", "scarlet",
]

FOODS = [
    "sushi", "tacos", "ramen", "pizza", "pad thai", "biryani", "paella",
    "goulash", "pho", "falafel", "dim sum", "ceviche", "pierogi", "bibimbap",
    "empanadas", "moussaka", "laksa", "jollof rice", "borscht", "shakshuka",
]

HOBBIES = [
    "painting", "hiking", "chess", "gardening", "pottery", "surfing",
    "birdwatching", "knitting", "rock climbing", "calligraphy",
    "woodworking", "sailing", "beekeeping", "origami", "astronomy",
    "fencing", "juggling", "bouldering", "archery", "foraging",
]


def _extract_relation(statement):
    """Extract the relation from a generated statement."""
    s = statement.lower()
    if " lives in " in s:
        return "lives in"
    if " works as " in s:
        return "works as"
    if "'s favorite color is " in s:
        return "favorite color is"
    if "'s favorite food is " in s:
        return "favorite food is"
    if " enjoys " in s:
        return "enjoys"
    return "is"


def _extract_object(fact):
    """Extract the object value from a fact dict (use expected keywords)."""
    return " ".join(fact["expected"])


def generate_facts(n, seed=42):
    """Generate n unique synthetic facts with statements and recall questions.

    Each fact is a single atomic piece of information:
      statement: "Elena Voronov is a software engineer."
      question:  "What does Elena Voronov do for work?"
      expected:  ["software engineer"]
    """
    rng = random.Random(seed)

    # Generate unique people
    people = []
    used_names = set()
    first_pool = list(FIRST_NAMES)
    last_pool = list(LAST_NAMES)
    rng.shuffle(first_pool)
    rng.shuffle(last_pool)

    for i in range(min(n, len(first_pool))):
        first = first_pool[i]
        last = last_pool[i % len(last_pool)]
        full = f"{first} {last}"
        if full not in used_names:
            used_names.add(full)
            people.append((first, last, full))

    # If we need more people than names, recombine
    while len(people) < n:
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        full = f"{first} {last}"
        if full not in used_names:
            used_names.add(full)
            people.append((first, last, full))

    facts = []
    rng.shuffle(CITIES)
    rng.shuffle(JOBS)
    rng.shuffle(PET_NAMES)
    rng.shuffle(COLORS)
    rng.shuffle(FOODS)
    rng.shuffle(HOBBIES)

    # Fact templates — each generates one atomic fact
    # raw_prompt: the completion prompt used for MEMIT recall testing
    templates = [
        lambda p, i: {
            "statement": f"{p[2]} lives in {CITIES[i % len(CITIES)]}.",
            "question": f"Where does {p[2]} live?",
            "raw_prompt": f"{p[2]} lives in",
            "expected": [CITIES[i % len(CITIES)]],
            "forbidden": [],
        },
        lambda p, i: {
            "statement": f"{p[2]} works as a {JOBS[i % len(JOBS)]}.",
            "question": f"What does {p[2]} do for work?",
            "raw_prompt": f"{p[2]} works as",
            "expected": [JOBS[i % len(JOBS)]],
            "forbidden": [],
        },
        lambda p, i: {
            "statement": f"{p[2]}'s favorite color is {COLORS[i % len(COLORS)]}.",
            "question": f"What is {p[2]}'s favorite color?",
            "raw_prompt": f"{p[2]}'s favorite color is",
            "expected": [COLORS[i % len(COLORS)]],
            "forbidden": [],
        },
        lambda p, i: {
            "statement": f"{p[2]}'s favorite food is {FOODS[i % len(FOODS)]}.",
            "question": f"What is {p[2]}'s favorite food?",
            "raw_prompt": f"{p[2]}'s favorite food is",
            "expected": [FOODS[i % len(FOODS)]],
            "forbidden": [],
        },
        lambda p, i: {
            "statement": f"{p[2]} enjoys {HOBBIES[i % len(HOBBIES)]} in their free time.",
            "question": f"What does {p[2]} do in their free time?",
            "raw_prompt": f"{p[2]} enjoys",
            "expected": [HOBBIES[i % len(HOBBIES)]],
            "forbidden": [],
        },
    ]

    for i in range(n):
        person = people[i % len(people)]
        template = templates[i % len(templates)]
        facts.append(template(person, i))

    return facts


def run_capacity_test(config_path, max_facts=100, batch_size=5, seed=42):
    """Inject facts in increments and measure recall at each level.

    Args:
        config_path: Config YAML path
        max_facts: Maximum facts to inject
        batch_size: Facts to inject per round before testing
        seed: Random seed for reproducible fact generation
    """
    start_time = time.time()

    config = Config(config_path)
    model_name = config.model["path"]

    print("=" * 70)
    print("  MEMIT CAPACITY TEST")
    print("=" * 70)
    print(f"  Model:      {model_name}")
    print(f"  Backend:    {config.model.get('backend', 'mlx')}")
    print(f"  Max facts:  {max_facts}")
    print(f"  Batch size: {batch_size}")
    print(f"  MEMIT layers: {config.get('memit.target_layers', [])}")
    print(f"  MEMIT lambda: {config.get('memit.lambda_reg', 0.5)}")
    print("=" * 70)
    print()

    # Generate all facts upfront
    all_facts = generate_facts(max_facts, seed=seed)

    # Initialize
    print("[INIT] Loading model...")
    orchestrator = Orchestrator(config)

    # Light reset
    import shutil
    for dir_key in ["training", "replay_buffer", "conversations"]:
        d = Path(config.paths.get(dir_key, ""))
        if d.exists():
            shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
    memit_dir = Path(config.paths.get("memit_data", "data/memit"))
    if memit_dir.exists():
        shutil.rmtree(memit_dir)
        memit_dir.mkdir(parents=True, exist_ok=True)
    orchestrator.edit_ledger.clear_all()
    orchestrator.chat.reset_turn_count()
    orchestrator.context.reset(keep_summary=False)

    # Disable auto-sleep/nap
    orchestrator.chat.set_sleep_callback(lambda t: None)
    orchestrator.chat.set_nap_callback(lambda t: None)

    print()

    results = {
        "config": {
            "model": model_name,
            "backend": config.model.get("backend", "mlx"),
            "memit_layers": config.get("memit.target_layers", []),
            "memit_lambda": config.get("memit.lambda_reg", 0.5),
            "max_facts": max_facts,
            "batch_size": batch_size,
        },
        "checkpoints": [],
    }

    injected_facts = []
    fact_idx = 0

    while fact_idx < max_facts:
        batch_end = min(fact_idx + batch_size, max_facts)
        batch = all_facts[fact_idx:batch_end]

        # Inject batch directly into MEMIT (bypass extractor — it only handles first-person)
        print(f"--- Injecting facts {fact_idx+1}-{batch_end} ---")
        triples = []
        for fact in batch:
            # Parse "Elena Voronov lives in Portland." → FactTriple
            stmt = fact["statement"]
            # Subject is the person's name (everything before the relation)
            for rel_phrase in [" lives in ", " works as ", "'s favorite color is ",
                               "'s favorite food is ", " enjoys "]:
                if rel_phrase in stmt or rel_phrase.lower() in stmt.lower():
                    subject = stmt.split(rel_phrase)[0].split("'s ")[0] if "'s " in rel_phrase else stmt.split(rel_phrase)[0]
                    break
            else:
                subject = stmt.split(" ")[0] + " " + stmt.split(" ")[1]
            triple = FactTriple(
                subject=subject.strip(),
                relation=_extract_relation(stmt),
                object=_extract_object(fact),
            )
            triples.append(triple)
            injected_facts.append(fact)

        if triples:
            edit = orchestrator.memit_engine.inject_facts(triples)
            edit_count = 1 if edit else 0
            print(f"  Injected {len(triples)} facts → {edit_count} MEMIT edit (batch)")

        fact_idx = batch_end
        total_injected = len(injected_facts)

        # Get MEMIT state
        status = orchestrator.get_status()
        memit_edits = status.get("memit_edits", 0)
        sleep_pressure = status.get("sleep_pressure", 0)

        # Test recall on ALL injected facts so far
        print(f"  Testing recall on {total_injected} facts (MEMIT edits: {memit_edits})...")

        # Reset context so the model can't use conversation history
        orchestrator.context.reset(keep_summary=False)
        orchestrator.chat.reset_turn_count()
        orchestrator.chat.set_sleep_callback(lambda t: None)
        orchestrator.chat.set_nap_callback(lambda t: None)

        passed = 0
        failed = 0
        partial = 0
        failed_facts = []

        for fact in injected_facts:
            # Use raw completion prompt (not question format) to match MEMIT injection.
            # MEMIT edits the completion pathway: "X lives in" → "Y"
            prompt = fact.get("raw_prompt", fact["question"])
            response = orchestrator.backend.generate(prompt, max_tokens=30)
            if response is None:
                response = ""

            resp_lower = response.lower()

            # Check expected keywords
            found_expected = sum(
                1 for kw in fact["expected"] if kw.lower() in resp_lower
            )
            total_expected = len(fact["expected"])

            # Check forbidden keywords
            found_forbidden = [
                kw for kw in fact.get("forbidden", []) if kw.lower() in resp_lower
            ]

            if found_expected == total_expected and not found_forbidden:
                passed += 1
            elif found_expected > 0:
                partial += 1
            else:
                failed += 1
                failed_facts.append({
                    "question": fact["question"],
                    "expected": fact["expected"],
                    "response": response[:100],
                })

        recall = passed / total_injected if total_injected else 0
        partial_rate = partial / total_injected if total_injected else 0

        checkpoint = {
            "total_facts": total_injected,
            "memit_edits": memit_edits,
            "sleep_pressure": round(sleep_pressure, 3),
            "passed": passed,
            "partial": partial,
            "failed": failed,
            "recall": round(recall, 3),
            "partial_rate": round(partial_rate, 3),
        }
        results["checkpoints"].append(checkpoint)

        print(f"  Facts: {total_injected} | MEMIT edits: {memit_edits} | "
              f"Pressure: {sleep_pressure:.3f}")
        print(f"  PASS: {passed}  PARTIAL: {partial}  FAIL: {failed}  "
              f"Recall: {recall:.2f}")

        if failed_facts:
            for ff in failed_facts[:3]:
                print(f"    FAIL: {ff['question']} expected={ff['expected']}")
                print(f"          got: {ff['response'][:80]}...")
            if len(failed_facts) > 3:
                print(f"    ... and {len(failed_facts)-3} more failures")

        print()

        # Stop early if recall drops below threshold
        if total_injected >= batch_size * 2 and recall < 0.3:
            print(f"  STOPPING EARLY — recall dropped to {recall:.2f}")
            break

        # Reset context again for next injection round
        orchestrator.context.reset(keep_summary=False)
        orchestrator.chat.reset_turn_count()
        orchestrator.chat.set_sleep_callback(lambda t: None)
        orchestrator.chat.set_nap_callback(lambda t: None)

    elapsed = time.time() - start_time
    results["elapsed_seconds"] = elapsed

    # Print summary
    print("=" * 70)
    print("  CAPACITY TEST SUMMARY")
    print("=" * 70)
    model_short = model_name.split("/")[-1]
    print(f"  Model: {model_short}")
    print()
    print(f"  {'Facts':>6} {'Edits':>6} {'Pass':>6} {'Part':>6} {'Fail':>6} {'Recall':>8} {'Pressure':>10}")
    print(f"  {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*10}")

    for cp in results["checkpoints"]:
        print(f"  {cp['total_facts']:>6} {cp['memit_edits']:>6} "
              f"{cp['passed']:>6} {cp['partial']:>6} {cp['failed']:>6} "
              f"{cp['recall']:>8.2f} {cp['sleep_pressure']:>10.3f}")

    # Find the degradation point
    peak_recall = 0
    peak_facts = 0
    degrade_point = None
    for cp in results["checkpoints"]:
        if cp["recall"] >= peak_recall:
            peak_recall = cp["recall"]
            peak_facts = cp["total_facts"]
        if degrade_point is None and cp["recall"] < 0.8 and cp["total_facts"] > batch_size:
            degrade_point = cp["total_facts"]

    results["peak_recall"] = peak_recall
    results["peak_facts"] = peak_facts
    results["degrade_point"] = degrade_point

    print()
    print(f"  Peak recall: {peak_recall:.2f} at {peak_facts} facts")
    if degrade_point:
        print(f"  Degradation (<0.80): starts at {degrade_point} facts")
    else:
        print(f"  No degradation detected (recall stayed >= 0.80)")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="MEMIT Capacity Test")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML")
    parser.add_argument("--max-facts", type=int, default=100,
                        help="Maximum facts to inject (default: 100)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Facts per batch before testing (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for fact generation")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    results = run_capacity_test(
        config_path=args.config,
        max_facts=args.max_facts,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        model_name = results["config"]["model"].split("/")[-1]
        output_path = Path("experiments/results") / f"{model_name}_capacity.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, float) and obj != obj:
            return None
        return obj

    with open(output_path, "w") as f:
        json.dump(clean(results), f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
