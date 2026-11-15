"""Visor tab "¿Cómo funciona?": what an individual is, who chooses the moves and what the 100
random scenarios change, with a live example of the loaded individual."""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QLabel, QSplitter,
                             QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget)

import battle_simulator as bs
from ga import operators as ops
from ui import gamedata as gd
from ui import theme
from ui.widgets import section_label

LOW_HP = 0.2  # "little HP left" in the example tables


def _explanation(genome, scores) -> str:
    """Explanatory text with the numbers of the loaded individual."""
    n_catch = sum(1 for p in genome[0] if p is not None)
    plans = genome[2] if len(genome) > 2 else []
    n_plan = sum(1 for plan in plans for x in (plan or []) if x is not None)
    plan_txt = (f"Este individuo asigna <b>{n_plan}</b> enfrentamientos." if n_plan
                else "Este individuo no tiene plan de cambios: cada Pokémon combate hasta debilitarse.")
    if scores is not None:
        fit = f"{scores.mean[0]:,.0f}".replace(",", ".")
        complete = f"{scores.p_complete[0] * 100:.0f}".replace(".", ",")
        scen_txt = (f"Este individuo tiene fitness <b>{fit}</b> (media de los 100) y gana los 21 combates en el "
                    f"<b>{complete} %</b> de los escenarios.")
    else:
        scen_txt = ("No se encuentra la tabla de tiradas (<code>python tools/precompute_damage.py --rolls</code>): "
                    "el visor muestra solo el simulador determinista.")
    return f"""
<h2>¿Qué es un individuo?</h2>
<p>Un individuo es una <b>ruta nuzlocke completa</b>. Su genotipo tiene tres partes, y
<b>ninguna dice qué ataque usar</b>:</p>
<ol>
<li><b>Capturas</b> (36 genes): qué Pokémon se captura en cada zona, o ninguno.
Este individuo captura <b>{n_catch}</b> de 36.</li>
<li><b>Equipos</b> (21 genes, uno por combate): qué Pokémon ya capturados se llevan (hasta 6) y
<b>en qué orden salen</b>. El primero empieza el combate.</li>
<li><b>Plan de cambios</b> (21 genes, uno por combate): para cada Pokémon del rival, qué miembro del
equipo debe enfrentarse a él, o "quien esté en el campo". {plan_txt}</li>
</ol>
<p>Es lo que decide el jugador <i>antes</i> de cada combate. Lo que pasa <i>dentro</i> del combate lo
deciden reglas fijas.</p>

<h2>¿Quién elige los ataques?</h2>
<p>Una regla fija, la misma para tu equipo y para el rival. Se aplica <b>en cada turno</b> según
<b>quién tiene delante y cuánta vida le queda</b>:</p>
<ol>
<li>Si algún ataque lo <b>debilita seguro</b> (incluso con la tirada de daño más baja), usa el más
preciso de esos; a igualdad, el de más prioridad.</li>
<li>Si no, el de <b>más daño esperado</b> por turno: precisión × daño medio con críticos × número de
golpes. Los ataques que tardan dos turnos (Rayo Solar, Excavar, Hiperrayo…) cuentan la mitad.</li>
</ol>
<p>Por eso no existe una secuencia del tipo "tres turnos Rayo Hielo y luego Surf". Si delante hay un
Dragonite usará Rayo Hielo; si sale un Charizard, Surf. Y si al rival le queda poca vida, puede usar
un ataque más débil que no falla. Las tablas de la derecha lo muestran para cualquier combate de este
individuo.</p>
<p>Excepciones: un ataque de dos turnos o un Golpe / Danza Pétalo / Enfado se termina aunque cambie
la situación, y después de Hiperrayo hay que recargar.</p>

<h2>¿Quién sale al campo?</h2>
<ul>
<li>Empieza el primero del equipo.</li>
<li>Cuando el rival saca un Pokémon que tiene asignado un miembro en el plan, y ese miembro está en pie
y no está ya en el campo, el individuo <b>gasta el turno en cambiar</b>. El cambio va primero y el rival
golpea al que entra.</li>
<li>Si tu Pokémon se debilita, sale gratis el asignado al rival actual si está en pie; si no, el siguiente
en el orden del equipo.</li>
</ul>

<h2>Los 100 escenarios</h2>
<p>Cada combate se juega <b>100 veces con el mismo individuo</b>: mismas capturas, mismos equipos, mismo
orden y mismo plan. Lo único que cambia es la suerte. Cada escenario es una secuencia fija de números
aleatorios que decide si un ataque acierta, si es crítico, la tirada de daño, los efectos secundarios,
si un Pokémon paralizado puede moverse…</p>
<p>Como cambia la suerte, cambian las situaciones: un fallo deja al rival con más vida, una parálisis
cambia quién ataca primero, un Pokémon cae antes y sale el siguiente… La regla de los ataques
<b>reacciona a cada situación</b>. Por eso el mismo individuo usa ataques distintos en escenarios
distintos.</p>
<p>El <b>fitness</b> es la media de los 100 escenarios: el % de HP perdido en los 21 combates, más 10000
por cada derrota. Los escenarios son siempre los mismos (los usan el entrenamiento y este visor), así
que un individuo tiene siempre el mismo fitness. {scen_txt} En la pestaña <i>Combates</i> se pueden ver
el peor escenario, el mediano y el mejor.</p>
<p>Con el <b>simulador determinista</b> de la memoria (<i>Escenarios aleatorios = Determinista</i>) no hay
suerte: tirada media, sin fallos, críticos ni efectos, y cada Pokémon usa siempre su ataque de más daño
contra el rival que tiene delante.</p>

<h2>¿Qué hace el algoritmo genético?</h2>
<p>Busca la mejor combinación de capturas, equipos, orden y plan de cambios:</p>
<ul>
<li><b>Cruza</b> individuos: el hijo hereda equipos completos de cada padre y se reparan las capturas
para que cumplan las reglas nuzlocke.</li>
<li><b>Muta</b> individuos: cambia una captura, un miembro de un equipo, el orden de salida o una entrada
del plan.</li>
<li><b>Se queda</b> con los de menor fitness, generación tras generación.</li>
</ul>
"""


class HowItWorksPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        lay.addWidget(splitter)

        self.text = QTextBrowser()
        self.text.setOpenLinks(False)
        self.text.document().setDefaultStyleSheet(
            f"h2 {{ color: {theme.YELLOW}; font-size: 16px; margin-top: 14px; }}"
            f"p, li {{ color: {theme.TEXT}; font-size: 13px; line-height: 140%; }}"
            f"b {{ color: white; }} i {{ color: {theme.MUTED}; }} code {{ color: {theme.ORANGE}; }}")
        splitter.addWidget(self.text)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(6, 0, 0, 0)
        rl.setSpacing(6)
        rl.addWidget(section_label("Ejemplo con este individuo"))
        row = QHBoxLayout()
        row.addWidget(QLabel("Combate:"))
        self.battle = QComboBox()
        for name in gd.TRAINER_NAMES:
            self.battle.addItem(name)
        row.addWidget(self.battle, 1)
        rl.addLayout(row)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.RichText)
        rl.addWidget(self.info)
        rl.addWidget(section_label("Qué ataque usa cada uno de tus Pokémon contra cada rival"))
        self.table_p = self._table()
        rl.addWidget(self.table_p, 1)
        rl.addWidget(section_label("Qué ataque usa cada rival contra cada uno de tus Pokémon"))
        self.table_r = self._table()
        rl.addWidget(self.table_r, 1)
        self.legend = QLabel("")
        self.legend.setWordWrap(True)
        self.legend.setProperty("role", "muted")
        rl.addWidget(self.legend)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 6)

        self.genome = None
        self.scores = None
        self.battle.currentIndexChanged.connect(self._refresh_example)

    @staticmethod
    def _table() -> QTableWidget:
        t = QTableWidget(0, 0)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionMode(QAbstractItemView.NoSelection)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        t.verticalHeader().setDefaultSectionSize(48)
        t.setWordWrap(True)
        return t

    def set_individual(self, genome, scores=None):
        self.genome, self.scores = genome, scores
        self.text.setHtml(_explanation(genome, scores))
        self._refresh_example()

    # ------------------------------------------------------------------ example

    def _store(self):
        try:
            import stochastic_battle as sb

            return sb.default_store()
        except (FileNotFoundError, KeyError):
            return None

    def _refresh_example(self, *_):
        if self.genome is None:
            return
        t = self.battle.currentIndex()
        genome = self.genome
        if not genome[1][t]:
            return
        store = self._store()
        rival = gd.rival_team(t, genome[0][0])
        base_team = list(genome[1][t])
        plan = genome[2][t] if len(genome) > 2 and genome[2][t] else []
        if store is not None:
            import stochastic_battle as sb

            spec = sb.compile_battle(genome, t, store)
            team, cap = spec.team, spec.level_cap
        else:
            spec = None
            team, cap = gd.evolved_team(genome, t), gd.level_cap(t, genome[0][0])

        order = " → ".join(f"{n + 1}. {gd.pkm_name(p)}" for n, p in enumerate(team))
        assigned = {}
        for j, pkm in enumerate(plan or []):
            if pkm is not None and pkm in base_team and j < len(rival):
                assigned[(base_team.index(pkm), j)] = True
        plan_txt = " · ".join(f"contra {rival[j][2]}: {gd.pkm_name(team[i])}" for i, j in assigned) or \
            "sin plan (cada Pokémon combate hasta debilitarse y sale el siguiente)"
        stats = ""
        if self.scores is not None:
            p = self.scores.win_rates[0, t] * 100
            hp = self.scores.battle_hp_lost[0, t]
            stats = (f"<br>En los 100 escenarios gana el <b>{p:.0f} %</b> de las veces y pierde de media "
                     f"<b>{hp:.0f} %</b> de HP.")
        self.info.setText(f"<b>Equipo (orden de salida):</b> {order}<br><b>Plan de cambios:</b> {plan_txt}{stats}")

        for table, side in ((self.table_p, 0), (self.table_r, 1)):
            table.clear()
            table.setRowCount(len(team))
            table.setColumnCount(len(rival))
            table.setHorizontalHeaderLabels([f"{r[2]} (Nv {r[1]})" for r in rival])
            table.setVerticalHeaderLabels([f"{n + 1}. {gd.pkm_name(p)} (Nv {cap})" for n, p in enumerate(team)])
            for i, pkm in enumerate(team):
                for j, r in enumerate(rival):
                    item = QTableWidgetItem(self._cell(store, spec, side, i, j, pkm, r, cap))
                    item.setTextAlignment(Qt.AlignCenter)
                    if (i, j) in assigned:
                        item.setBackground(QColor(255, 203, 5, 45))
                        item.setToolTip(f"Plan de cambios: {gd.pkm_name(pkm)} se enfrenta a {r[2]}")
                    table.setItem(i, j, item)
        if store is not None:
            self.legend.setText(
                f"Primera línea: ataque elegido con el defensor a toda su vida. Segunda línea: si le queda poca vida "
                f"({LOW_HP:.0%}). El motivo es la regla: «debilita seguro» o «más daño esperado». En amarillo, los "
                "enfrentamientos del plan de cambios. Durante el combate la elección depende de la vida que le "
                "quede de verdad al defensor en cada turno de cada escenario.")
        else:
            self.legend.setText("Simulador determinista: cada Pokémon usa siempre su ataque de más daño contra el "
                                "rival que tiene delante, le quede la vida que le quede.")

    def _cell(self, store, spec, side, i, j, pkm, r, cap) -> str:
        if store is None:  # deterministic table: always the most damaging move
            name = bs.dataset["pkdex"][str(pkm)]["name"]
            key = (bs.damage_key(name, cap, r[2], r[1], False) if side == 0
                   else bs.damage_key(r[2], r[1], name, cap, True))
            entry = bs.dataset["damage_dictionary"].get(key)
            return gd.format_move(entry[0]) if entry else "-"
        import stochastic_battle as sb

        key_id = int(spec.keys[i, j, side])
        max_hp = int(spec.r_max[j] if side == 0 else spec.p_max[i])
        full, why_full = sb.explain_choice(store, key_id, max_hp)
        low, why_low = sb.explain_choice(store, key_id, max(1, int(max_hp * LOW_HP)))
        return f"{gd.format_move(full)} ({why_full})\ncon poca vida: {gd.format_move(low)} ({why_low})"
