# rag/knowledge_base.py
# Farm Management RAG Knowledge Base
#
# Stores agronomic best-practice documents, crop thresholds, and intervention
# protocols that ground the LLM's recommendations in verified knowledge rather
# than training-set approximations.
#
# Production replacement: swap KnowledgeBase with ChromaDB + sentence-transformers.
# The RAGRetriever.retrieve() interface stays identical.

import math, re
from typing import List, Tuple, Dict


class Chunk:
    __slots__ = ("cid", "category", "title", "text", "_vec")
    def __init__(self, cid, category, title, text):
        self.cid = cid; self.category = category
        self.title = title; self.text = text; self._vec = {}


CORPUS: List[Chunk] = [

    Chunk("SOIL-01", "soil", "Soil Moisture Critical Thresholds",
        """Optimal soil moisture for crop production varies by crop and growth stage.
Wheat: 40-60% field capacity; below 20% causes wilting stress; above 80% causes root anoxia.
Corn: 50-70% field capacity; critical during silking (75-80%); tolerates short dry spells in vegetative.
Soybeans: 50-70% field capacity; most sensitive during pod-fill (R3-R6 stages).
Irrigation trigger: initiate when soil moisture drops below 40% field capacity for sensitive stages.
Over-irrigation risk: above 80% moisture suppresses nitrogen uptake and promotes root disease."""),

    Chunk("SOIL-02", "soil", "Soil pH and Nutrient Availability",
        """Optimal soil pH ranges: wheat 6.0-7.0, corn 5.8-7.0, soybeans 6.0-6.8.
Below pH 5.5: aluminum and manganese toxicity; phosphorus becomes unavailable.
Above pH 7.5: iron, manganese, zinc, and boron deficiency become likely.
Nitrogen: most available between pH 6.0-8.0. Nitrate leaching risk above field capacity.
Phosphorus: peak availability pH 6.0-7.0. Tie-up below 5.5 and above 7.5.
Potassium: available across broad pH range; deficiency common in sandy soils.
Organic matter above 3%: significantly improves water holding capacity and nutrient cycling."""),

    Chunk("SOIL-03", "soil", "Compaction and Tillage Guidelines",
        """Soil compaction index above 50 indicates significant restriction to root growth.
Compaction above 70: emergency sub-soiling required before next planting season.
Compaction causes: heavy equipment on wet soil, repeated tillage at same depth.
Remediation: deep tillage (chisel plow or subsoiler) when soil is dry enough.
Avoid tillage when soil moisture exceeds 60% field capacity — creates compaction pans.
No-till benefit: reduces surface compaction; improves organic matter over 3-5 years."""),

    Chunk("CROP-01", "crop_management", "Crop Growth Stage Decision Points",
        """Wheat growth stages (Feekes scale):
  Feekes 1-5 (tillering): maintain moisture; apply nitrogen at Feekes 4-5.
  Feekes 6-9 (stem extension): critical irrigation window; protect from disease.
  Feekes 10-11 (heading/grain fill): highest water demand; fungicide application window.
Corn growth stages (V/R notation):
  V1-V6: establish stand; weed control critical; modest water needs.
  V7-VT: rapid growth; nitrogen side-dress by V6; escalate irrigation.
  R1 (silking): peak water demand — 0.5 inch/day; drought here cuts yield 50%.
  R3-R6 (kernel fill): maintain moisture; fungicide if disease pressure high.
Soybean growth stages:
  V1-V5: canopy closure goal; weed management window closes at V3.
  R1-R2 (flowering): begin irrigation; protect from foliar diseases.
  R3-R5 (pod fill): highest water demand; key yield-determining period."""),

    Chunk("CROP-02", "crop_management", "Pest and Disease Intervention Thresholds",
        """Economic thresholds for intervention (treat when pest exceeds threshold):
Wheat: aphids >100 per stem at boot stage; Hessian fly — use resistant varieties.
Corn: rootworm >1 beetle/plant; European corn borer >1 egg mass per 5 plants at VT.
Soybeans: bean leaf beetle >2/plant vegetative; soybean aphid >250/plant at R1-R4.
Disease spray thresholds:
  Wheat: fusarium head blight risk model > 0.5 at heading — apply prothioconazole.
  Corn: gray leaf spot or northern corn leaf blight exceeding 5% leaf area — apply strobilurin.
  Soybeans: frogeye leaf spot > 5 plants/10 ft row at R3 stage.
High disease risk triggers: >10 consecutive hours leaf wetness, temperature 20-30C."""),

    Chunk("IRR-01", "irrigation", "Irrigation Scheduling Best Practices",
        """Evapotranspiration-based scheduling (ET method) is the recommended standard.
Daily crop water use (peak season): wheat 0.2-0.3 in/day, corn 0.25-0.35 in/day, soybeans 0.2-0.3 in/day.
Drip/micro-irrigation efficiency: 90-95%. Sprinkler: 75-85%. Flood/furrow: 50-70%.
Timing: early morning irrigation reduces evaporation losses by 20-30%.
Deficit irrigation: apply 70-80% ETc during vegetative stages to encourage deep rooting.
Full replacement: critical reproductive stages (silking, pod-fill) require 100% ETc replacement.
Avoid irrigation within 48 hours of forecast rainfall >10mm — risk of waterlogging."""),

    Chunk("IRR-02", "irrigation", "Waterlogging and Drainage",
        """Waterlogging (soil moisture >80%) causes oxygen depletion in root zone within 24-48 hours.
Corn most sensitive: yield loss begins after 2 days ponding at V6-VT; 50% loss after 4 days.
Soybeans: tolerates 1-2 days at vegetative; 20% yield loss per additional day at flowering.
Wheat: dormant wheat tolerates waterlogging better than actively growing wheat.
Drainage actions: stop all irrigation immediately; open tile drain outlets if available.
Post-waterlogging scouting: check for pythium root rot and nitrogen leaching loss."""),

    Chunk("FERT-01", "fertilization", "Nitrogen Management Guidelines",
        """Wheat nitrogen requirements: total 120-180 lbs N/acre depending on yield goal.
  Apply 30-40 lbs N/acre pre-plant; top-dress remainder at Feekes 4-5.
Corn nitrogen: 1.2 lbs N per bushel yield goal. 200 bu/ac goal = 240 lbs N/acre.
  Split application: 30% pre-plant, 70% side-dress at V4-V6 reduces leaching risk.
Soybeans: fix 50-70% of own nitrogen via Bradyrhizobium; supplement 20-30 lbs N/acre if nodulation poor.
Nitrogen soil test interpretation:
  < 20 ppm: definite deficiency — apply full recommendation immediately.
  20-50 ppm: marginal — apply 50% of standard rate.
  > 50 ppm: adequate — delay application; re-test in 4 weeks."""),

    Chunk("HARVEST-01", "harvest", "Harvest Timing and Quality Management",
        """Wheat harvest: optimal at 13-14% grain moisture; above 16% requires drying (costly).
  Test weight target: >60 lbs/bushel. Below 56 lbs/bu = Grade 4 — significant discount.
  Harvest window closes when grain moisture drops below 11% — shattering losses increase.
Corn harvest: optimal 18-22% grain moisture for storage; field dry to 14-15%.
  Each 1% moisture above 14% costs $0.03-0.05/bushel for commercial drying.
  Stalk quality deterioration begins 3-4 weeks after black layer — prioritize weak-stalk fields.
Soybeans: harvest at 12-14% moisture. Below 10% causes significant shattering losses.
  Pod shatter risk: varieties with shattering scores >4 should be prioritised in harvest order.
Weather window needed: 3+ consecutive dry days (< 5mm forecast) with relative humidity < 70%."""),

    Chunk("MARKET-01", "market", "Commodity Price Decision Rules",
        """Forward contracting guidelines for row crop farms:
Wheat: consider forward contract when price >$7.00/bushel (historical profitable threshold Central Valley).
Corn: profitable to contract at >$5.50/bushel when input costs are at current levels.
Soybeans: contract 20-30% of expected bushels at > $13.50/bushel to lock in profitability.
Basis risk: local cash price typically $0.20-0.50 below CME futures.
Harvest pressure: cash prices typically drop 10-15% at peak harvest — pre-harvest contracting advised.
Storage premium: soybeans and corn typically gain $0.05-0.08/month in storage Jan-May."""),
]


class KnowledgeBase:
    _STOP = {"a","an","the","and","or","in","on","at","to","for","of","with","by",
             "is","are","was","were","be","it","per","not","if","as","from","when"}

    def __init__(self):
        self._idf: Dict[str, float] = {}
        self._build()

    def _tok(self, t: str) -> List[str]:
        return [w for w in re.findall(r"[a-z0-9%./\-]+", t.lower())
                if w not in self._STOP and len(w) > 1]

    def _build(self):
        N = len(CORPUS)
        df: Dict[str, int] = {}
        for c in CORPUS:
            for t in set(self._tok(c.title + " " + c.text)):
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log(N / v) for t, v in df.items()}
        for c in CORPUS:
            toks = self._tok(c.title + " " + c.text)
            n = len(toks) or 1
            tf: Dict[str, float] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            c._vec = {t: (cnt / n) * self._idf.get(t, 0) for t, cnt in tf.items()}

    def _qvec(self, q: str) -> Dict[str, float]:
        toks = self._tok(q); n = len(toks) or 1
        tf: Dict[str, float] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        return {t: (cnt / n) * self._idf.get(t, 0) for t, cnt in tf.items()}

    @staticmethod
    def _cos(v1, v2) -> float:
        dot = sum(v1[t] * v2[t] for t in v1 if t in v2)
        m1 = math.sqrt(sum(x * x for x in v1.values()))
        m2 = math.sqrt(sum(x * x for x in v2.values()))
        return dot / (m1 * m2 + 1e-9)

    def retrieve(self, query: str, top_k: int = 3,
                 category: str = None) -> List[Tuple[Chunk, float]]:
        qv = self._qvec(query)
        pool = [c for c in CORPUS if category is None or c.category == category]
        scored = sorted([(c, self._cos(qv, c._vec)) for c in pool], key=lambda x: -x[1])
        return [(c, s) for c, s in scored[:top_k] if s > 0.0]

    def retrieve_text(self, query: str, top_k: int = 3, category: str = None) -> str:
        results = self.retrieve(query, top_k, category)
        if not results:
            return f"[RAG] No documents found for: {query}"
        lines = [f"[RAG — {len(results)} doc(s) for: '{query}']"]
        for i, (c, s) in enumerate(results, 1):
            lines.append(f"\n[{i}] {c.title}  (category: {c.category}, score: {s:.3f})")
            lines.append(c.text)
        return "\n".join(lines)


FARM_KB = KnowledgeBase()
