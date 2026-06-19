# Modelo de la vista cenital (BEV / IPM)

> Documenta el planteamiento geométrico que hay detrás de `perception/bev.py`:
> cómo, a partir de **una sola foto del dron**, reconstruimos una **vista cenital
> métrica** del suelo (bird's-eye view). Sin redes neuronales — solo geometría
> proyectiva.

---

## 1. Planteamiento del problema

Tenemos una cámara montada en el dron a una **altura conocida `H`**, mirando
aproximadamente en horizontal (pitch ≈ 0, roll ≈ 0). Queremos responder a:

> *Dado un píxel `(u, v)` de la imagen, ¿a qué punto `(X, Y)` del suelo
> corresponde, en metros?*

Y a la inversa: *dado un punto del suelo, ¿en qué píxel cae?* Si sabemos hacer
ese mapeo, podemos **reproyectar la parte del suelo de la foto sobre una rejilla
cenital métrica** (un pseudo-ortofoto). A esto se le llama **Inverse Perspective
Mapping (IPM)**.

La hipótesis clave que lo hace posible es el **supuesto de mundo plano**: lo que
vemos es el suelo, el plano `Z = 0`. Bajo esa hipótesis, el mapa de un plano del
mundo a la imagen es una **homografía** (matriz 3×3), y todo se vuelve álgebra
lineal.

---

## 2. La cámara como *pinhole*

Modelamos la cámara con el modelo **pinhole** (estenopeico), el modelo matemático
más simple de una cámara:

- Existe un **único punto óptico** (el "agujero").
- Toda la luz entra por ese punto.
- Los puntos 3D del mundo se **proyectan sobre un plano de imagen**.

No hay lente, ni distorsión, ni profundidad de campo: solo proyección central. Es
una idealización, pero es suficiente para nuestra geometría.

---

## 3. La matriz de intrínsecos `K`

La matriz `K` describe cómo la cámara proyecta el espacio 3D `(X, Y, Z)` (en
coordenadas de cámara) a píxeles `(u, v)`. Es el **estándar absoluto de la visión
por computador** desde hace décadas; no viene de un paper concreto, sino del
propio modelo pinhole y de la geometría proyectiva.

### Derivación

**1. Proyección sobre el plano imagen.** Un punto 3D en coordenadas de cámara
`(X, Y, Z)` se proyecta dividiendo por la profundidad:

```
x = X / Z
y = Y / Z
```

**2. De coordenadas físicas a píxeles.** Escalamos por la distancia focal en
píxeles (`fx`, `fy`) y desplazamos al origen de la imagen (`cx`, `cy`):

```
u = fx · x + cx
v = fy · y + cy
```

**3. Forma matricial.** Juntando todo en coordenadas homogéneas aparece `K`:

```
        ⎡ fx   0   cx ⎤
  K  =  ⎢  0  fy   cy ⎥
        ⎣  0   0    1 ⎦

  s · [u, v, 1]ᵀ  =  K · [X, Y, Z]ᵀ
```

donde `s` es un factor de escala (la profundidad) que se cancela en la división
proyectiva.

### Significado de cada parámetro

| Parámetro | Qué es |
|-----------|--------|
| `fx`, `fy` | distancia focal en píxeles (horizontal / vertical) |
| `cx`, `cy` | *principal point*: dónde el eje óptico cruza el sensor |

---

## 4. Aproximar `K` a partir del FOV

No conocemos la **calibración real** de la cámara del Tello / RoboMaster TT. Lo
que sí conocemos es la **resolución** y el **campo de visión (FOV)**. Con eso
construimos una `K` aproximada — la pregunta que respondemos es:

> *"¿Qué matriz `K` tendría una cámara pinhole ideal con este FOV y esta
> resolución?"*

Es una técnica completamente estándar. Las focales salen de la trigonometría del
FOV:

```
fx = (W / 2) / tan(HFOV / 2)
fy = (H / 2) / tan(VFOV / 2)
```

donde:

- `W` = ancho de la imagen en píxeles
- `H` = alto de la imagen en píxeles
- `HFOV` = campo de visión horizontal
- `VFOV` = campo de visión vertical

En el código (`CameraModel.from_fov`), si no se proporciona `VFOV` se asume
**píxeles cuadrados** (`fy = fx`), que es lo correcto cuando solo confías en el
dato horizontal. El `VFOV` implícito se deriva con:

```
VFOV = 2 · atan( tan(HFOV / 2) · H / W )
```

### FOV del Tello / RoboMaster TT

La especificación nominal de la cámara es **82.6°**, que es el **FOV diagonal**
(DJI no aclara horizontal vs diagonal, pero el valor encaja como diagonal). Con la
resolución de foto `2592 × 1936` derivamos el horizontal y el vertical reales:

```
HFOV = 2 · atan( (W / d) · tan(DFOV / 2) ),   d = √(W² + H²)
```

que da:

- `HFOV ≈ 70.3°`
- `VFOV ≈ 55.5°`

Esto está implementado en `hfov_from_dfov(...)`: tomamos el 82.6° como diagonal y
obtenemos el HFOV horizontal con el que se construye `K`.

> **Aproximación vs. rigor.** Derivar `HFOV`/`VFOV` desde el FOV diagonal y el
> *aspect ratio* (asumiendo píxeles cuadrados y *principal point* centrado) es una
> **aproximación rápida**: nos basta una cifra de catálogo y la resolución, sin
> tocar la cámara. Se podría ir más allá y ser más riguroso con una **calibración
> con tablero de ajedrez** (`cv2.calibrateCamera`), que estima directamente
> `fx`, `fy`, `cx`, `cy` **y los coeficientes de distorsión** del objetivo a
> partir de varias fotos del patrón. **De momento no lo estamos haciendo** — nos
> quedamos en la aproximación por FOV.

---

## 5. ¿Cómo obtenemos `cx` y `cy`?

`cx` y `cy` son el *principal point*: el punto donde el eje óptico atraviesa el
sensor. En una cámara perfectamente alineada eso es el **centro de la imagen**.

Como **no tenemos calibración real**, hacemos exactamente esa suposición —
principal point centrado:

```
cx = W / 2
cy = H / 2
```

Así aparece en el código (`CameraModel.from_fov`):

```python
cx, cy = width / 2.0, height / 2.0
```

En una cámara real, `cx`/`cy` se desvían un poco del centro por imperfecciones de
montaje del sensor, y se obtendrían mediante una **calibración** (p. ej. tablero
de ajedrez con `cv2.calibrateCamera`). Para nuestro caso no calibrado, asumir el
centro es la aproximación razonable y estándar.

---

## 6. ¿Para qué sirve la altura `H`?

`K`, `pitch` y `roll` fijan la **forma** de la proyección: a qué *dirección* del
mundo apunta cada píxel. Pero una dirección no es una distancia. Lo que convierte
esa dirección en metros sobre el suelo es la **altura del dron `H`**:

- En **`pixels_to_ground`**, el rayo de cada píxel se intersecta con el suelo en
  `t = −H / d_z`. La altura es la distancia al plano `Z = 0`, así que es lo que
  dice *cuán lejos* viaja el rayo hasta tocar el suelo.
- En **`ground_homography`**, `H` entra en la columna `−H · c3` de la homografía
  suelo→imagen.

Consecuencia práctica: **`H` es lo único que da escala métrica absoluta a la
BEV**, y la escala es **lineal** en `H`. Si la altura está mal por un factor 2,
toda la rejilla métrica queda mal por ese factor 2 (un objeto a 3 m se mediría a
6 m). Por eso interesa leer la altura **real** en el momento de la foto —
preferiblemente del sensor ToF (distancia hacia abajo) y, si no, la barométrica—
en lugar de un valor fijo. Eso es lo que hace `_height_m_from_meta(...)` a partir
del *sidecar* JSON del snapshot.

---

## 7. De `K` a la vista cenital (resumen del pipeline)

Con `K` (intrínsecos) y la orientación de la cámara (extrínsecos: `pitch`,
`roll`, altura `H`) ya podemos cerrar el problema:

1. **Suelo → imagen** (`ground_homography`): homografía 3×3 que lleva un punto
   métrico del suelo `(X, Y)` a un píxel `(u, v)`.
2. **Imagen → suelo** (`pixels_to_ground`): el camino inverso, por intersección
   rayo/plano (`t = −H / d_z`); descarta los rayos que apuntan por encima del
   horizonte.
3. **`BEVProjector`**: precalcula, para cada celda de una rejilla métrica cenital,
   qué píxel de origen le corresponde (mapeo *hacia atrás*, para no dejar
   agujeros), y luego reproyecta cualquier frame con `cv2.remap`.

Como el mapa suelo↔imagen es una homografía, el IPM analítico y
`cv2.warpPerspective` son la **misma transformación** — el módulo lo verifica
numéricamente en `compare_methods`.

---

## 8. La máscara de suelo (`ground_mask`)

La máscara marca, sobre la imagen **original**, qué píxeles son "suelo
aprovechable" para la BEV. Punto clave: **no mira el contenido de la imagen**
(ni color, ni textura, ni detección) — la define **pura geometría** a partir de
la pose supuesta (`H`, `pitch`, `roll`) y de `K`. Para cada píxel se pregunta *"¿a
dónde va su rayo y dónde aterriza?"* (vía `pixels_to_ground`), y entra en la
máscara si cumple **dos condiciones**:

1. **El rayo baja por debajo del horizonte.** La dirección del rayo tiene una
   componente vertical `d_z`. Si `d_z < 0` el rayo desciende y cruza el suelo
   `Z = 0` a distancia `t = −H / d_z > 0` (suelo candidato); si `d_z ≥ 0` apunta
   al horizonte o al cielo y **nunca toca el suelo** (`valid = False`). El
   **horizonte es exactamente `d_z = 0`** — con pitch=0 cae en la fila central
   (`v = cy`): todo lo de abajo es suelo geométrico, lo de arriba no.
2. **El impacto cae dentro de la ventana métrica de la BEV.** El punto `(X, Y)`
   donde el rayo toca el suelo debe estar en el rectángulo cubierto por la BEV:

   ```
   in_range = valid
              & x_range[0] ≤ X ≤ x_range[1]
              & y_range[0] ≤ Y ≤ y_range[1]
   ```

El resultado es una máscara `uint8` del tamaño de la imagen (`255` = suelo útil).
En la demo CLI se muestra como el panel central de `original | ground mask | BEV`.

> Igual que en el resto del módulo, "suelo" es **geométrico, no semántico**: una
> pared u obstáculo vertical dentro del rango también se marca como suelo, porque
> asumimos mundo plano. Distinguir suelo real de un obstáculo requeriría
> profundidad o detección, que aquí no hay.

---

## 9. Referencias

1. Richard Hartley, Andrew Zisserman.
   *Multiple View Geometry in Computer Vision*.
   2nd Edition, Cambridge University Press, 2003.
   ISBN: 978-0521540513.

   *(Referencia clásica del modelo pinhole, la matriz `K` y la geometría
   proyectiva. Prácticamente toda la visión por computador moderna usa esta
   formulación.)*
