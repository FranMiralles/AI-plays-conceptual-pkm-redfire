# Animaciones de movimientos por tipo

El visor de combates (`ui/effects.py`) usa estas animaciones: cuando un Pokémon ataca, se reproduce sobre el objetivo la animación del tipo elemental del movimiento.

Son pixel art original generado por código con `python tools/generate_move_effects.py`, al estilo de las animaciones de la GBA:
- celdas de 64×64 px;
- paletas cortas;
- píxeles duros, sin suavizado;
- contorno oscuro de 1 px.

Cada PNG es una tira con los fotogramas de izquierda a derecha, y `animations.json` indica cuántos fotogramas tiene cada tipo. El visor los amplía a un múltiplo entero para que los píxeles queden nítidos.

| Tipo | Animación |
|---|---|
| Normal | Destello de impacto |
| Lucha | Golpe de kárate y estallido |
| Volador | Ráfaga de viento con plumas |
| Veneno | Burbujas tóxicas que estallan |
| Tierra | Grieta, polvo y rocas que saltan |
| Roca | Rocas que caen y se rompen |
| Bicho | Agujas convergentes |
| Fantasma | Bola sombra que estalla |
| Acero | Zarpazos metálicos con destello |
| Fuego | Llamas que crecen y ascuas |
| Agua | Salpicadura en corona y gotas |
| Planta | Hojas giratorias y cortes |
| Eléctrico | Rayos y chispas |
| Psíquico | Ondas concéntricas |
| Hielo | Cristales que se forman y se rompen |
| Dragón | Llamas dracónicas en espiral |
| Siniestro | Mandíbulas que muerden |

Para modificar una animación, edita su función en `tools/generate_move_effects.py` y vuelve a ejecutar el script.
