import express from "express";
import { calculate, Pokemon, Move, Generations, toID } from "@smogon/calc";

const app = express();
app.use(express.json({ limit: "20mb" }));

// moveOptions (opcional): { isCrit, hits } para golpes críticos o el daño de un solo golpe
function runCalc(gen, attacker, defender, move, moveOptions) {
  return calculate(
    gen,
    new Pokemon(gen, attacker.species, {
      level: attacker.level,
      ability: attacker.ability,
      item: attacker.item,
      nature: "Hardy", // Naturaleza neutra, IVs al máximo por defecto, sin EVs
      boosts: attacker.boosts, // opcional: fases de características ({ atk: -1, ... })
      status: attacker.status, // opcional: "brn", "par"...
    }),
    new Pokemon(gen, defender.species, {
      level: defender.level,
      ability: defender.ability,
      item: defender.item,
      nature: "Hardy",
      boosts: defender.boosts,
      status: defender.status,
    }),
    new Move(gen, move, moveOptions || {})
  );
}

app.get("/health", (_req, res) => res.json({ ok: true }));

app.post("/calc", (req, res) => {
  try {
    const { genReq, attacker, defender, move } = req.body;
    res.json(runCalc(Generations.get(genReq), attacker, defender, move));
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Varios cálculos en una sola petición: devuelve [{damage, maxHP}] o [{error}] por elemento
app.post("/calc_batch", (req, res) => {
  const { genReq, items } = req.body;
  const gen = Generations.get(genReq);
  res.json(
    items.map(({ attacker, defender, move, moveOptions }) => {
      try {
        const result = runCalc(gen, attacker, defender, move, moveOptions);
        return { damage: result.damage, maxHP: result.defender.rawStats.hp };
      } catch (error) {
        return { error: error.message };
      }
    })
  );
});

// Datos de 3.ª generación para recalcular el daño con cambios de características y estados:
// estadísticas base, tipos, peso y habilidades de las especies; potencia, tipo, categoría y
// contacto de los movimientos; y la tabla de tipos.
app.post("/gen_data", (req, res) => {
  try {
    const { genReq, species, moves } = req.body;
    const gen = Generations.get(genReq);
    const outSpecies = {};
    for (const name of species || []) {
      const sp = gen.species.get(toID(name));
      if (!sp) { outSpecies[name] = null; continue; }
      outSpecies[name] = { name: sp.name, baseStats: sp.baseStats, types: sp.types, weightkg: sp.weightkg,
                           abilities: sp.abilities };
    }
    const outMoves = {};
    for (const name of moves || []) {
      try {
        const mv = new Move(gen, name);
        outMoves[name] = { name: mv.name, bp: mv.bp, type: mv.type, category: mv.category,
                           contact: !!(mv.flags && mv.flags.contact), priority: mv.priority,
                           multihit: mv.multihit || null, secondaries: !!mv.secondaries };
      } catch (error) {
        outMoves[name] = null;
      }
    }
    const types = {};
    for (const t of gen.types) {
      types[t.name] = t.effectiveness;
    }
    res.json({ species: outSpecies, moves: outMoves, types });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`API de damage calculator en http://localhost:${PORT}`);
});
