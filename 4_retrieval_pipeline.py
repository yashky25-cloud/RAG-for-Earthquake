import re
import pandas as pd

INPUT = "pipeline1_evidence.txt"
OUTPUT = "pipeline3_importance.csv"


# ============================================================
# 1. CANONICAL PARAMETER RULES
# ============================================================

def canonicalize(p):

    p = re.sub(r"\s+", " ", p.lower().strip())

    # --------------------------------------------------------
    # THROUGH / BOND STONES
    # --------------------------------------------------------

    if any(x in p for x in [
        "through stone",
        "through stones",
        "bond stone",
        "bond stones",
        "bonding stone",
        "bonding stones",
        "absence of through stones",
        "lack of through stones"
    ]):
        return "Absence of Through/Bond Stones"


    # --------------------------------------------------------
    # CORNERS / JUNCTIONS
    # --------------------------------------------------------

    if any(x in p for x in [
        "corner",
        "corners",
        "quoins",
        "junction",
        "junctions",
        "corner bond",
        "corner bonding",
        "corner reinforcement"
    ]):
        return "Weak Corners/Junctions"


    # --------------------------------------------------------
    # MORTAR
    # --------------------------------------------------------

    if any(x in p for x in [
        "mortar",
        "mud mortar",
        "cement mortar",
        "lime mortar",
        "mortar joint",
        "mortar joints",
        "mortar quality",
        "mortar deterioration",
        "mortar failure",
        "weak mortar",
        "poor mortar",
        "inadequate mortar",
        "silt mortar"
    ]):
        return "Weak Masonry Mortar"


    # --------------------------------------------------------
    # UNREINFORCED MASONRY
    # --------------------------------------------------------

    if any(x in p for x in [
        "unreinforced",
        "un-reinforced",
        "absence of reinforcement",
        "lack of reinforcement",
        "without reinforcement",
        "no reinforcement"
    ]):
        return "Unreinforced Masonry"


    # --------------------------------------------------------
    # MASONRY BONDING
    # --------------------------------------------------------

    if any(x in p for x in [
        "poor masonry bond",
        "weak masonry bond",
        "poor masonry bonding",
        "weak masonry bonding",
        "inadequate masonry bond",
        "inadequate masonry bonding",
        "poor wall bonding",
        "weak wall bonding",
        "poor interlocking",
        "weak interlocking",
        "inadequate interlocking",
        "poorly bonded masonry",
        "poorly bonded wall"
    ]):
        return "Poor Masonry Bonding"


    # --------------------------------------------------------
    # HEAVY ROOF
    # --------------------------------------------------------

    if any(x in p for x in [
        "heavy roof",
        "heavy roofing",
        "heavy roof load",
        "heavy roof covering",
        "heavy roof mass",
        "large roof mass",
        "large mass roof",
        "heavy stone roof",
        "heavy tile roof"
    ]):
        return "Heavy Roof Load"


    # --------------------------------------------------------
    # LIGHT / TIMBER ROOF
    # --------------------------------------------------------

    if any(x in p for x in [
        "light roof",
        "light roofing",
        "lightweight roof",
        "lightweight roofing",
        "thatch",
        "thatch roof",
        "timber roof",
        "timber roofing",
        "timber roof system",
        "timber reinforcement",
        "timber band",
        "timber bands"
    ]):
        return "Light Roof/Timber Reinforcement"


    # --------------------------------------------------------
    # WALL-ROOF CONNECTION
    # --------------------------------------------------------

    if any(x in p for x in [
        "wall plate",
        "wall-plate",
        "wall roof connection",
        "wall-roof connection",
        "roof wall connection",
        "roof-wall connection",
        "wall roof restraint",
        "wall-roof restraint",
        "roof wall restraint",
        "roof-wall restraint",
        "roof anchorage",
        "poor roof anchorage",
        "weak roof anchorage",
        "roof not tied",
        "roof not anchored",
        "binding of roof with wall",
        "binding of roof with walls"
    ]):
        return "Weak Wall-Roof Restraint"


    # --------------------------------------------------------
    # ROOF TRUSSES
    # --------------------------------------------------------

    if "truss" in p:
        return "Lack of Roof Trusses"


    # --------------------------------------------------------
    # ROOF COVERING / ELEMENTS
    # --------------------------------------------------------

    if any(x in p for x in [
        "roof covering",
        "roof cover",
        "roofing material",
        "roofing materials",
        "slate roof",
        "slate roofing",
        "gi sheet",
        "gi sheets",
        "corrugated sheet",
        "corrugated sheets",
        "roof sheet",
        "roof sheets",
        "tile roof",
        "tile roofing",
        "roof element",
        "roof elements",
        "roof"
    ]):
        return "Roof Covering/Elements"


    # --------------------------------------------------------
    # FOUNDATION / SITE
    # --------------------------------------------------------

    if any(x in p for x in [
        "foundation",
        "foundations",
        "retaining wall",
        "retaining walls",
        "topography",
        "topographic",
        "slope",
        "sloping site",
        "steep slope",
        "site instability",
        "ground instability",
        "foundation instability",
        "ground condition",
        "soil condition"
    ]):
        return "Foundation/Site Instability"


    # --------------------------------------------------------
    # WALL ORIENTATION / GEOMETRY
    # --------------------------------------------------------

    if any(x in p for x in [
        "orientation",
        "oriented",
        "aligned",
        "alignment",
        "direction",
        "directionality",
        "trending",
        "geometry",
        "geometric",
        "irregular wall",
        "wall irregularity",
        "plan irregularity",
        "vertical irregularity"
    ]):
        return "Wall Orientation/Geometry"


    # --------------------------------------------------------
    # OPENINGS / PROJECTIONS
    # --------------------------------------------------------

    if any(x in p for x in [
        "opening",
        "openings",
        "door opening",
        "window opening",
        "large opening",
        "irregular opening",
        "stairs",
        "staircase",
        "projection",
        "projections"
    ]):
        return "Openings/Projections"


    # --------------------------------------------------------
    # WALL MASS / STIFFNESS
    # --------------------------------------------------------

    if any(x in p for x in [
        "wall mass",
        "mass of wall",
        "wall stiffness",
        "stiffness of wall",
        "massive wall"
    ]):
        return "Wall Mass/Stiffness"


    # --------------------------------------------------------
    # PARTITION WALL
    # --------------------------------------------------------

    if any(x in p for x in [
        "partition wall",
        "partition walls",
        "partition vulnerability",
        "weak partition",
        "partition failure",
        "partition damage"
    ]):
        return "Partition Wall Vulnerability"


    # --------------------------------------------------------
    # GABLE WALL
    # --------------------------------------------------------

    if any(x in p for x in [
        "gable wall",
        "gable walls",
        "gable vulnerability",
        "gable failure",
        "gable damage"
    ]):
        return "Gable Wall Vulnerability"


    # --------------------------------------------------------
    # MASONRY AGE / CONDITION
    # --------------------------------------------------------

    if any(x in p for x in [
        "aged masonry",
        "aging masonry",
        "ageing masonry",
        "old masonry",
        "old wall",
        "old walls",
        "deteriorated masonry",
        "deterioration of masonry",
        "weathered masonry",
        "weathering of masonry",
        "masonry ageing"
    ]):
        return "Masonry Ageing"


    # --------------------------------------------------------
    # PLASTER
    # --------------------------------------------------------

    if "plaster" in p:
        return "Masonry/Plaster Condition"


    # --------------------------------------------------------
    # GENERAL MASONRY
    #
    # IMPORTANT:
    # Broad enough to prevent excessive "Other",
    # but does NOT classify every occurrence of "masonry".
    # --------------------------------------------------------

    if any(x in p for x in [
        "poor masonry",
        "weak masonry",
        "poor quality masonry",
        "weak quality masonry",
        "poor masonry construction",
        "poor construction of masonry",
        "poorly constructed masonry",
        "weak masonry construction",
        "inadequate masonry construction",
        "poor workmanship",
        "poor construction quality",
        "poor construction",
        "weak wall",
        "poor wall",
        "weak stone masonry",
        "poor stone masonry",
        "weak brick masonry",
        "poor brick masonry",
        "rubble masonry",
        "random rubble",
        "random rubble masonry",
        "field stone masonry",
        "mud stone masonry"
    ]):
        return "Poor/Weak Masonry"


    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return "Other"


# ============================================================
# 2. DAMAGE SCORE
# ============================================================

def damage_score(damage):

    d = damage.lower().strip()

    # --------------------------------------------------------
    # 0 = NO DAMAGE
    # --------------------------------------------------------

    if any(x in d for x in [
        "without any damage",
        "without damage",
        "no damage",
        "no significant damage",
        "escaped without damage",
        "withstood without damage"
    ]):
        return 0


    # --------------------------------------------------------
    # 5 = COMPLETE DESTRUCTION
    # --------------------------------------------------------

    if any(x in d for x in [
        "total collapse",
        "total collapsed",
        "complete collapse",
        "completely collapsed",
        "totally collapsed",
        "caved in",
        "destroyed",
        "complete destruction"
    ]):
        return 5


    # --------------------------------------------------------
    # 4 = SEVERE DAMAGE / FAILURE
    # --------------------------------------------------------

    if any(x in d for x in [
        "partial collapse",
        "partially collapsed",
        "partial roof collapse",
        "extensively damaged",
        "extensive damage",
        "severely damaged",
        "severe damage",
        "severely cracked",
        "severe cracking",
        "major damage",
        "major failure",
        "structural failure",
        "wall failure",
        "roof failure",
        "failure of",
        "failed",
        "collapsed",
        "collapse"
    ]):
        return 4


    # --------------------------------------------------------
    # 3 = SIGNIFICANT / MODERATE DAMAGE
    # --------------------------------------------------------

    if any(x in d for x in [
        "deep cracks",
        "deep cracking",
        "open cracks",
        "open cracking",
        "large cracks",
        "large cracking",
        "shear cracks",
        "shear cracking",
        "extensive cracks",
        "extensive cracking",
        "bulging",
        "displacement",
        "displaced",
        "significant damage",
        "significant cracking",
        "moderate damage",
        "moderately damaged",
        "unsatisfactory behaviour",
        "unsatisfactory performance"
    ]):
        return 3


    # --------------------------------------------------------
    # 2 = GENERAL DAMAGE
    # --------------------------------------------------------

    if any(x in d for x in [
        "damage",
        "damaged",
        "cracking",
        "cracks",
        "cracked",
        "tilting",
        "tilt",
        "minor displacement"
    ]):
        return 2


    # --------------------------------------------------------
    # 1 = MINOR DAMAGE
    # --------------------------------------------------------

    if any(x in d for x in [
        "minor",
        "hairline cracks",
        "hair fine cracks",
        "fine cracks",
        "peeling",
        "slight tilt",
        "slight tilting",
        "slight bulging",
        "low level of damage",
        "relatively less damage",
        "minor damage"
    ]):
        return 1


    return 1


# ============================================================
# 3. READ PIPELINE 1
# ============================================================

with open(INPUT, "r", encoding="utf-8") as f:
    text = f.read()


pattern = re.compile(
    r'Parameter:\s*(.*?)\s*'
    r'Damage:\s*(.*?)\s*'
    r'Evidence:\s*"(.*?)"',
    re.DOTALL
)

records = pattern.findall(text)


data = []

for parameter, damage, evidence in records:

    data.append({
        "Original_Parameter": parameter.strip(),
        "Canonical_Parameter": canonicalize(parameter),
        "Damage": damage.strip(),
        "Damage_Score": damage_score(damage),
        "Evidence": evidence.strip()
    })


df = pd.DataFrame(data)


# ============================================================
# 4. REMOVE "OTHER"
# ============================================================
# Other is not treated as a final vulnerability parameter.
# Its records are excluded from weight calculation.
# ============================================================

df_valid = df[
    df["Canonical_Parameter"] != "Other"
].copy()


# ============================================================
# 5. PARAMETER FREQUENCY + DAMAGE
# ============================================================

result = (
    df_valid
    .groupby("Canonical_Parameter")
    .agg(
        Evidence_Frequency=("Canonical_Parameter", "size"),
        Damage_Total=("Damage_Score", "sum")
    )
    .reset_index()
)


# ============================================================
# 6. NORMALIZE FREQUENCY AND DAMAGE
# ============================================================

result["Frequency_Factor"] = (
    result["Evidence_Frequency"] /
    result["Evidence_Frequency"].max()
)

result["Damage_Factor"] = (
    result["Damage_Total"] /
    result["Damage_Total"].max()
)


# ============================================================
# 7. RAW IMPORTANCE
# ============================================================
# Frequency = 40%
# Damage    = 60%
# ============================================================

result["Raw_Importance"] = (
    0.40 * result["Frequency_Factor"] +
    0.60 * result["Damage_Factor"]
)


# ============================================================
# 8. FINAL IMPORTANCE SCALE
# ============================================================
# Final Importance Factor is scaled to 0.50 - 0.90.
#
# This is ONLY a presentation/weight scaling step.
# Ranking is determined from Raw_Importance.
# ============================================================

raw_min = result["Raw_Importance"].min()
raw_max = result["Raw_Importance"].max()

if raw_max == raw_min:

    result["Importance_Factor"] = 0.90

else:

    result["Importance_Factor"] = (
        0.50 +
        (
            (result["Raw_Importance"] - raw_min) /
            (raw_max - raw_min)
        ) * 0.40
    )


# ============================================================
# 9. ROUND FACTORS TO 2 DECIMAL PLACES
# ============================================================

result["Frequency_Factor"] = (
    result["Frequency_Factor"].round(2)
)

result["Damage_Factor"] = (
    result["Damage_Factor"].round(2)
)

result["Importance_Factor"] = (
    result["Importance_Factor"].round(2)
)


# ============================================================
# 10. RANK
# ============================================================

result["Rank"] = (
    result["Raw_Importance"]
    .rank(
        ascending=False,
        method="dense"
    )
    .astype(int)
)


# ============================================================
# 11. SORT
# ============================================================

result = result.sort_values(
    by=[
        "Raw_Importance",
        "Damage_Total",
        "Evidence_Frequency"
    ],
    ascending=[False, False, False]
)


# ============================================================
# 12. FINAL COLUMNS
# ============================================================

result = result[
    [
        "Canonical_Parameter",
        "Evidence_Frequency",
        "Damage_Total",
        "Frequency_Factor",
        "Damage_Factor",
        "Importance_Factor",
        "Rank"
    ]
]


# ============================================================
# 13. SAVE
# ============================================================

result.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 14. OUTPUT
# ============================================================

print("\n================ FINAL PARAMETERS ================\n")

print(
    result.to_string(index=False)
)

print("\n============================================")
print("Raw records              :", len(df))
print("Excluded 'Other' records :", len(df) - len(df_valid))
print("Valid evidence records   :", len(df_valid))
print("Canonical params         :", result.shape[0])
print("Output                   :", OUTPUT)
print("============================================")