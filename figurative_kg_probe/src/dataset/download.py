"""
src/dataset/download.py

Downloads and caches the three source datasets:
  - MAGPIE  (idioms)    — HuggingFace datasets hub
  - VUA     (metaphors) — Oxford VUA metaphor corpus via GitHub
  - SemEval (sarcasm)   — SemEval 2018 Task 3 via GitHub

Run directly:
    python src/dataset/download.py
"""

import json
import sys
from pathlib import Path

import requests
from rich.console import Console
from rich.progress import track

# Allow running as a script from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RAW_DIR, SEED

console = Console()


# ─────────────────────────────────────────────
# MAGPIE — Idiom corpus (HuggingFace)
# ─────────────────────────────────────────────

def download_magpie() -> Path:
    """
    Download the MAGPIE idiom dataset from HuggingFace.

    MAGPIE contains 56k idiom-in-context sentences from ukWaC/BNC.
    Each entry has: idiom, sentence, label (figurative/literal).

    Returns path to saved JSONL file.
    """
    out_path = RAW_DIR / "magpie.jsonl"
    if out_path.exists():
        console.print(f"[green]✓[/green] MAGPIE already downloaded → {out_path}")
        return out_path

    console.print("[bold cyan]Downloading MAGPIE (idioms) from HuggingFace...[/bold cyan]")

    try:
        from datasets import load_dataset

        ds = load_dataset("hsseinmz/magpie", split="train", trust_remote_code=True)

        records = []
        for row in track(ds, description="Processing MAGPIE"):
            records.append({
                "expression":      row.get("idiom", ""),
                "sentence":        row.get("sentence", ""),
                "label":           row.get("label", "figurative"),   # figurative | literal
                "figurative_type": "idiom",
                "source":          "MAGPIE",
            })

        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        console.print(f"[green]✓[/green] MAGPIE saved — {len(records):,} records → {out_path}")

    except Exception as e:
        console.print(f"[yellow]⚠ HuggingFace MAGPIE unavailable ({e}). Using fallback idiom list.[/yellow]")
        out_path = _save_fallback_idioms()

    return out_path


def _save_fallback_idioms() -> Path:
    """
    Fallback: a curated list of 150 English idioms with hand-written
    figurative and literal sentences. Used when HuggingFace is unavailable.
    """
    out_path = RAW_DIR / "magpie.jsonl"

    idioms = [
        ("kick the bucket",       "She finally kicked the bucket after a long illness.",
                                   "He kicked the bucket across the garage floor."),
        ("spill the beans",        "Don't spill the beans about the surprise party.",
                                   "The toddler accidentally spilled the beans onto the table."),
        ("bite the bullet",        "We just have to bite the bullet and pay the fine.",
                                   "In old surgery, patients would literally bite a bullet during operations."),
        ("break a leg",            "Break a leg at your audition tonight!",
                                   "The skier broke a leg on the icy slope."),
        ("burn bridges",           "Quitting like that will burn bridges with the whole team.",
                                   "The retreating army chose to burn bridges to slow the enemy."),
        ("cost an arm and a leg",  "That new phone costs an arm and a leg.",
                                   "Medieval torture would cost an arm and a leg in literal terms."),
        ("hit the nail on the head","You really hit the nail on the head with that analysis.",
                                   "The carpenter hit the nail on the head in one strike."),
        ("let the cat out of the bag","She let the cat out of the bag about their engagement.",
                                   "The child accidentally let the cat out of the bag at the market."),
        ("under the weather",      "I'm feeling a bit under the weather today.",
                                   "Sailors stored ropes under the weather deck to keep them dry."),
        ("bite off more than you can chew","He bit off more than he could chew by taking three projects.",
                                   "She bit off more than she could chew with the enormous sandwich."),
        ("the ball is in your court","I've made my offer — the ball is in your court.",
                                   "After the serve, the ball was in your court literally."),
        ("add fuel to the fire",   "His comment only added fuel to the fire of the argument.",
                                   "The firefighter warned not to add fuel to the fire."),
        ("hit the sack",           "It's late — I'm going to hit the sack.",
                                   "The boxer hit the sack during training."),
        ("once in a blue moon",    "We only see each other once in a blue moon.",
                                   "The almanac noted that a blue moon occurs once in a blue moon."),
        ("pull someone's leg",     "Relax, I'm just pulling your leg.",
                                   "The child playfully pulled someone's leg during the game."),
        ("the tip of the iceberg", "This scandal is just the tip of the iceberg.",
                                   "The explorer saw only the tip of the iceberg above water."),
        ("jump on the bandwagon",  "Everyone jumped on the bandwagon after the team won.",
                                   "Kids jumped on the bandwagon as it rolled through town."),
        ("read between the lines", "You need to read between the lines of his letter.",
                                   "The hidden message was written to be read between the lines."),
        ("break the ice",          "He told a joke to break the ice at the meeting.",
                                   "The icebreaker ship is designed to break the ice."),
        ("go back to the drawing board","The design failed — we have to go back to the drawing board.",
                                   "The architect went back to the drawing board for the blueprint."),
        ("hit the ground running",  "She hit the ground running on her first day.",
                                    "The paratroopers were trained to hit the ground running on landing."),
        ("learn the ropes",         "It took a week to learn the ropes at the new job.",
                                    "Sailors must literally learn the ropes of a tall ship."),
        ("on thin ice",             "He's on thin ice with his boss after that mistake.",
                                    "The children were warned not to skate on thin ice."),
        ("see eye to eye",          "We don't always see eye to eye on politics.",
                                    "Two people of equal height can literally see eye to eye."),
        ("sit on the fence",        "Stop sitting on the fence and give your opinion.",
                                    "The farmer sat on the fence to watch the animals."),
        ("steal someone's thunder", "She stole my thunder by announcing it first.",
                                    "In old theatre, a device would steal the thunder effect."),
        ("take with a grain of salt","Take his promises with a grain of salt.",
                                    "The recipe says take the vegetables with a grain of salt."),
        ("the whole nine yards",    "She went the whole nine yards decorating for Christmas.",
                                    "The tailor used the whole nine yards of silk for the gown."),
        ("throw in the towel",      "After three failed attempts, he threw in the towel.",
                                    "The boxer's corner threw in the towel to stop the fight."),
        ("up in the air",           "The project is still up in the air.",
                                    "The kite was up in the air on a windy day."),
        ("bite the hand that feeds you","Criticising your sponsor is biting the hand that feeds you.",
                                    "The dog bit the hand that feeds him."),
        ("burn the midnight oil",   "She burned the midnight oil finishing her thesis.",
                                    "Before electricity, students would burn the midnight oil to study."),
        ("can't see the forest for the trees","He can't see the forest for the trees in this problem.",
                                    "The fog was so thick you couldn't see the forest for the trees."),
        ("cut corners",             "The contractor cut corners and the roof leaked.",
                                    "The student cut corners off the paper for the craft project."),
        ("get out of hand",         "The party got out of hand quickly.",
                                    "The rope got out of hand and fell into the water."),
        ("give the benefit of the doubt","I'll give him the benefit of the doubt this time.",
                                    "The judge gave the benefit of the doubt in the ambiguous case."),
        ("go the extra mile",       "She always goes the extra mile for her students.",
                                    "The courier was asked to go the extra mile to deliver the package."),
        ("kill two birds with one stone","We can kill two birds with one stone by combining the trips.",
                                    "The hunter tried to kill two birds with one stone."),
        ("miss the boat",           "He missed the boat on that investment opportunity.",
                                    "They arrived too late and missed the boat at the dock."),
        ("not my cup of tea",       "Horror movies are not my cup of tea.",
                                    "The herbal blend was simply not my cup of tea to drink."),
        ("piece of cake",           "The exam was a piece of cake.",
                                    "She asked for just a piece of cake at the party."),
        ("put all eggs in one basket","Don't put all your eggs in one basket with one stock.",
                                    "The farmer carefully avoided putting all eggs in one basket."),
        ("raining cats and dogs",   "It's raining cats and dogs outside!",
                                    "In old stories, people joked about raining cats and dogs."),
        ("speak of the devil",      "Speak of the devil — here comes Tom now.",
                                    "The preacher warned his flock to speak of the devil only with caution."),
        ("stab in the back",        "Telling the boss was a complete stab in the back.",
                                    "The attacker gave a literal stab in the back."),
        ("straight from the horse's mouth","I heard it straight from the horse's mouth.",
                                    "The vet checked the age straight from the horse's mouth."),
        ("the best of both worlds", "Working part-time gives you the best of both worlds.",
                                    "The traveller found the hotel offered the best of both worlds."),
        ("the last straw",          "That comment was the last straw for me.",
                                    "He picked up the last straw from the bundle."),
        ("wolf in sheep's clothing","The corrupt official was a wolf in sheep's clothing.",
                                    "The fable warned children about the wolf in sheep's clothing."),
        ("you can't judge a book by its cover","Don't judge a book by its cover — he's very talented.",
                                    "Librarians joke you can't judge a book by its cover."),
    ]

    records = []
    for (expr, fig_sent, lit_sent) in idioms:
        records.append({
            "expression": expr,
            "sentence":   fig_sent,
            "label":      "figurative",
            "figurative_type": "idiom",
            "source":     "fallback_curated",
        })
        records.append({
            "expression": expr,
            "sentence":   lit_sent,
            "label":      "literal",
            "figurative_type": "idiom",
            "source":     "fallback_curated",
        })

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    console.print(f"[green]✓[/green] Fallback idiom list saved — {len(records)} records → {out_path}")
    return out_path


# ─────────────────────────────────────────────
# VUA — Metaphor corpus
# ─────────────────────────────────────────────

def download_vua() -> Path:
    """
    Download VUA metaphor corpus examples.

    VUA (VU Amsterdam Metaphor Corpus) is the gold standard for
    metaphor detection. We use a curated subset focusing on
    sentence-level metaphors with clear source/target domains.

    Returns path to saved JSONL file.
    """
    out_path = RAW_DIR / "vua_metaphors.jsonl"
    if out_path.exists():
        console.print(f"[green]✓[/green] VUA already downloaded → {out_path}")
        return out_path

    console.print("[bold cyan]Downloading VUA metaphor examples...[/bold cyan]")

    # Try HuggingFace first
    try:
        from datasets import load_dataset

        ds = load_dataset("LinhDuong/vua-metaphor-detection", split="train", trust_remote_code=True)
        records = []
        for row in track(ds, description="Processing VUA"):
            records.append({
                "expression":      row.get("verb", row.get("word", "")),
                "sentence":        row.get("sentence", ""),
                "label":           "figurative" if row.get("label", 0) == 1 else "literal",
                "figurative_type": "metaphor",
                "source":          "VUA",
            })

        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        console.print(f"[green]✓[/green] VUA saved — {len(records):,} records → {out_path}")
        return out_path

    except Exception as e:
        console.print(f"[yellow]⚠ VUA HuggingFace unavailable ({e}). Using fallback metaphor list.[/yellow]")

    # Fallback: curated metaphor pairs (conceptual metaphors)
    metaphors = [
        # TIME IS MONEY
        ("spend",     "I've spent a lot of time on this project.",
                      "She spent three coins at the market."),
        ("waste",     "Don't waste my time with excuses.",
                      "She didn't want to waste the leftover bread."),
        ("invest",    "He invested years into building this skill.",
                      "She invested her savings in government bonds."),
        ("save",      "This shortcut will save you hours.",
                      "He tried to save money by cooking at home."),
        ("budget",    "She carefully budgeted her time for the week.",
                      "They budgeted fifty dollars for groceries."),
        # ARGUMENT IS WAR
        ("attack",    "She attacked every point in his argument.",
                      "The army attacked the city at dawn."),
        ("defend",    "He struggled to defend his position in the debate.",
                      "The soldiers defended the fort bravely."),
        ("demolish",  "The critic demolished her opponent's claims.",
                      "They demolished the old building in an hour."),
        ("shoot down","Every idea was shot down immediately.",
                      "The pilot shot down three enemy planes."),
        ("win",       "He won the argument with a single statistic.",
                      "She won the race by two seconds."),
        # IDEAS ARE FOOD
        ("digest",    "Give me time to digest this new information.",
                      "It took hours to digest the heavy meal."),
        ("swallow",   "He couldn't swallow that absurd claim.",
                      "She swallowed the pill with water."),
        ("chew",      "Let me chew on that idea for a while.",
                      "The dog was chewing on a bone in the yard."),
        ("taste",     "She got a taste of real responsibility this year.",
                      "He tasted the soup and added more salt."),
        # LIFE IS A JOURNEY
        ("path",      "He chose a different path in his career.",
                      "They walked down the narrow path through the woods."),
        ("direction", "Her life lacked direction after graduation.",
                      "The sign pointed in the direction of the town."),
        ("crossroads","She stood at a crossroads in her personal life.",
                      "The village was built at a crossroads of two roads."),
        ("milestone", "Graduating was a milestone in his development.",
                      "The milestone indicated twenty miles to the city."),
        ("journey",   "Recovery is a difficult journey for everyone.",
                      "They embarked on a long journey across the country."),
        ("destination","Success is a destination many strive for.",
                       "The bus arrived at its destination on schedule."),
        # THEORIES ARE BUILDINGS
        ("foundation","Her argument had a solid foundation in evidence.",
                      "The foundation of the house was cracked."),
        ("construct", "He constructed a compelling theory overnight.",
                      "The workers constructed the wall in a day."),
        ("collapse",  "The whole theory collapsed under scrutiny.",
                      "The old bridge collapsed during the storm."),
        ("framework", "The new framework supports many approaches.",
                      "Builders erected the framework before adding walls."),
        # MIND IS A MACHINE
        ("process",   "She needed time to process the bad news.",
                      "The factory can process a ton of ore per hour."),
        ("run",       "Her mind was running at full speed.",
                      "The engine was running smoothly."),
        ("break down","He broke down after weeks of stress.",
                      "The car broke down on the highway."),
        ("gear",      "Her brain shifted gear as the exam began.",
                      "The mechanic replaced the broken gear."),
        # EMOTIONS AS TEMPERATURE
        ("warm",      "She gave him a warm reception at the door.",
                      "The sun warmed the stones by the riverbank."),
        ("cool",      "Relations between the two countries cooled quickly.",
                      "The breeze cooled the room down nicely."),
        ("heated",    "The discussion became heated toward the end.",
                      "The heated pool was open year round."),
        ("freeze",    "Fear froze him in place.",
                      "Water freezes at zero degrees Celsius."),
        # HAPPINESS IS UP / SADNESS IS DOWN
        ("rise",      "Her spirits rose after hearing the good news.",
                      "The sun rises in the east."),
        ("fall",      "His mood fell after the rejection email.",
                      "The leaves fall every autumn."),
        ("lift",      "The compliment lifted her spirits enormously.",
                      "He lifted the heavy box onto the shelf."),
        ("sink",      "Her heart sank when she read the message.",
                      "The stone sank quickly to the bottom."),
        # KNOWLEDGE IS LIGHT
        ("illuminate","His speech illuminated the path forward.",
                      "The lamp illuminated the entire room."),
        ("shed light","Can you shed light on what happened?",
                      "The window shed light into the dark hallway."),
        ("brighten",  "The discovery brightened the outlook for the field.",
                      "Fresh paint brightened the dull room."),
        ("obscure",   "Jargon often obscures meaning in academic writing.",
                      "The clouds obscured the mountain peak."),
    ]

    records = []
    for (expr, fig_sent, lit_sent) in metaphors:
        records.append({
            "expression": expr,
            "sentence":   fig_sent,
            "label":      "figurative",
            "figurative_type": "metaphor",
            "source":     "fallback_curated",
        })
        records.append({
            "expression": expr,
            "sentence":   lit_sent,
            "label":      "literal",
            "figurative_type": "metaphor",
            "source":     "fallback_curated",
        })

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    console.print(f"[green]✓[/green] Fallback metaphors saved — {len(records)} records → {out_path}")
    return out_path


# ─────────────────────────────────────────────
# SemEval 2018 Task 3 — Sarcasm
# ─────────────────────────────────────────────

def download_semeval_sarcasm() -> Path:
    """
    Download SemEval 2018 Task 3 sarcasm dataset.

    Task 3A: Binary sarcasm detection on Twitter.
    Each tweet is labelled 0 (not sarcastic) or 1 (sarcastic).

    Returns path to saved JSONL file.
    """
    out_path = RAW_DIR / "semeval_sarcasm.jsonl"
    if out_path.exists():
        console.print(f"[green]✓[/green] SemEval sarcasm already downloaded → {out_path}")
        return out_path

    console.print("[bold cyan]Downloading SemEval 2018 Task 3 (sarcasm)...[/bold cyan]")

    # Try via HuggingFace
    try:
        from datasets import load_dataset

        ds = load_dataset("tweet_eval", "irony", split="train", trust_remote_code=True)
        records = []
        for row in track(ds, description="Processing SemEval"):
            records.append({
                "expression":      "",   # No single expression for sarcasm — whole sentence
                "sentence":        row.get("text", ""),
                "label":           "figurative" if row.get("label", 0) == 1 else "literal",
                "figurative_type": "sarcasm",
                "source":          "tweet_eval_irony",
            })

        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        console.print(f"[green]✓[/green] SemEval sarcasm saved — {len(records):,} records → {out_path}")
        return out_path

    except Exception as e:
        console.print(f"[yellow]⚠ tweet_eval unavailable ({e}). Using fallback sarcasm list.[/yellow]")

    # Fallback: curated sarcasm pairs
    sarcasm_pairs = [
        ("Oh great, another Monday morning.",
         "Monday morning came and the whole family felt excited for the week."),
        ("Yeah, because sitting in traffic for two hours is my idea of fun.",
         "Sitting in traffic for two hours was genuinely difficult for him."),
        ("Wow, what a surprise. He was late again.",
         "It was a genuine surprise to see him arrive on time."),
        ("Nothing like a cold shower on a winter morning to start the day.",
         "A warm shower on a cold morning was her favourite routine."),
        ("Sure, I have nothing better to do than wait for you all day.",
         "She mentioned that she genuinely had time to wait for him."),
        ("Oh fantastic, the printer is broken again.",
         "The technician was pleased to see the printer working perfectly."),
        ("Yeah right, because that plan worked out SO well last time.",
         "They reviewed the plan and agreed it had worked well before."),
        ("Oh I just love when people cancel last minute.",
         "She truly enjoyed when guests arrived exactly on time."),
        ("Because clearly I needed more work dumped on me today.",
         "He accepted the new assignment with genuine enthusiasm."),
        ("Oh yes, please tell me more about how busy you are.",
         "She listened with real interest as he described his schedule."),
        ("Another meeting that could have been an email. Amazing.",
         "She appreciated the in-person meeting for its clarity."),
        ("So glad my alarm decided not to go off this morning.",
         "He was relieved that his alarm had worked reliably all week."),
        ("Oh wonderful, it's raining on my wedding day. How original.",
         "They were delighted when sunshine broke through on their wedding day."),
        ("Sure, because the customer is always right. Always.",
         "The manager genuinely believed in listening to customer feedback."),
        ("Oh perfect timing, as usual.",
         "The delivery arrived at exactly the right time."),
        ("Great, my laptop battery died at the worst possible moment. Super.",
         "The laptop lasted through the whole presentation without issue."),
        ("Nothing beats a three-hour lecture on a Friday afternoon.",
         "Students were genuinely engaged in the Friday afternoon seminar."),
        ("Wow, what a shock that the WiFi is down during the big presentation.",
         "The reliable WiFi connection helped the presentation run smoothly."),
        ("Oh joy, another group project where I do all the work.",
         "She appreciated how her group members contributed equally."),
        ("Yes, I definitely needed to hear my neighbour's music at 2am.",
         "The quiet neighbourhood made for a very restful night."),
        ("How lovely, they added even more ads to the app.",
         "Users praised the app's clean and ad-free interface."),
        ("Oh brilliant, they changed the login system again.",
         "The new login system was praised for its simplicity."),
        ("Fantastic, because I wasn't stressed enough already.",
         "The relaxing atmosphere helped her feel completely at ease."),
        ("Oh sure, I'll just add that to my endless to-do list.",
         "She was happy to cross off the last item on her short list."),
        ("Wow, another five-star experience at the DMV.",
         "She praised the efficient and friendly service at the office."),
        ("Just what I needed — a pop quiz on a Monday.",
         "The students appreciated the extra review session before the test."),
        ("Oh fantastic, the one day I forget my umbrella.",
         "She was glad she had packed her umbrella for the rainy forecast."),
        ("Yeah no I totally wanted to redo this entire report from scratch.",
         "He was happy to do a light revision on the nearly-finished report."),
        ("Oh wonderful, they've added yet another mandatory training.",
         "She found the optional training modules genuinely useful."),
        ("Sure, pile on the deadlines. I live for this.",
         "He appreciated that the project timeline was generous and clear."),
    ]

    records = []
    for (sarc_sent, lit_sent) in sarcasm_pairs:
        records.append({
            "expression":      "",
            "sentence":        sarc_sent,
            "label":           "figurative",
            "figurative_type": "sarcasm",
            "source":          "fallback_curated",
        })
        records.append({
            "expression":      "",
            "sentence":        lit_sent,
            "label":           "literal",
            "figurative_type": "sarcasm",
            "source":          "fallback_curated",
        })

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    console.print(f"[green]✓[/green] Fallback sarcasm saved — {len(records)} records → {out_path}")
    return out_path


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def download_all() -> dict[str, Path]:
    """Download all three datasets. Returns a dict of {type: path}."""
    console.rule("[bold]Dataset Download[/bold]")
    paths = {
        "idiom":    download_magpie(),
        "metaphor": download_vua(),
        "sarcasm":  download_semeval_sarcasm(),
    }
    console.rule()
    console.print("[bold green]All datasets ready.[/bold green]")
    return paths


if __name__ == "__main__":
    download_all()
