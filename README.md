# FTHA Optimization

Modelo numérico do ciclo Otto com adição de calor em tempo finito (FTHA,
*Finite-Time Heat Addition*), baseado no trabalho de Naaktgeboren (2017). O
projeto teve origem em um estudo da disciplina de Máquinas Térmicas do curso de
Engenharia Mecânica da UTFPR, campus Guarapuava, e prepara o modelo
termodinâmico para uso posterior em otimização multiobjetivo.

A implementação reutilizável encontra-se em [`src/FTHA.py`](src/FTHA.py).

## Modelo

O ciclo preserva as hipóteses ar-padrão e de gás ideal do ciclo Otto, mas
substitui a adição instantânea de calor a volume constante por uma liberação de
calor com duração angular finita. Assim, o modelo incorpora:

- a geometria biela-manivela;
- a posição angular do virabrequim;
- o instante de ignição;
- a duração da adição de calor;
- o histórico acumulado de liberação de calor;
- calores específicos dependentes da temperatura.

Compressão, adição de calor e expansão são discretizadas em subprocessos
politrópicos. A rejeição de calor que fecha o ciclo permanece isocórica. O
sistema é fechado, internamente reversível e sem atrito, vazamentos ou perdas de
pressão; portanto, o modelo é apropriado para análise termodinâmica, mas não
substitui uma simulação de combustão ou um modelo preditivo de motor real.

### Geometria biela-manivela

Para um cilindro com volume deslocado unitário $V_{d,u}$, volume de folga
$V_c$ e taxa de compressão $r$,

$$
V_c=\frac{V_{d,u}}{r-1}.
$$

No motor quadrado usado no artigo, $D=S$, com raio da manivela $R=S/2$ e
comprimento da biela $L$. O deslocamento do pistão a partir do PMS é

$$
x(\alpha)=R(1-\cos\alpha)
+L\left[1-\sqrt{1-\left(\frac{R}{L}\right)^2\sin^2\alpha}\right],
$$

e o volume instantâneo é

$$
V(\alpha)=V_c+\frac{\pi D^2}{4}x(\alpha).
$$

Nesta implementação, \(\alpha=0\) corresponde ao ponto morto superior (PMS) e
\(\alpha=\pm\pi\) ao ponto morto inferior (PMI).

### Adição de calor

Se $N$ é a rotação em rpm e \(\Delta t_c\) é o tempo de combustão, a duração
angular da adição de calor é

$$
\omega=\frac{2\pi N}{60},
\qquad
\delta=\omega\Delta t_c.
$$

Para uma ignição iniciada em \(\theta\), a fração acumulada de calor é modelada
por

$$
y(\alpha)=
\begin{cases}
0, & \alpha<\theta,\\[4pt]
\dfrac{1}{2}-\dfrac{1}{2}
\cos\!\left[\dfrac{\pi(\alpha-\theta)}{\delta}\right],
& \theta\leq\alpha\leq\theta+\delta,\\[8pt]
1, & \alpha>\theta+\delta.
\end{cases}
$$

O calor específico fornecido no intervalo $i\rightarrow i+1$ é

$$
q_i=\left[y(\alpha_{i+1})-y(\alpha_i)\right]q_{in}.
$$

### Propriedades e solução numérica

O fluido obedece a $Pv=R_gT$. Os coeficientes dos polinômios de terceiro grau
para $c_p(T)$, armazenados em [`data/data.csv`](data/data.csv), são usados por
[`src/gas_prop.py`](src/gas_prop.py) para calcular $c_v(T)$, energia interna,
temperatura, pressão e volume específico.

Cada intervalo de volume variável é tratado como um subprocesso politrópico,

$$
Pv^{n_i}=C_i,
$$

com trabalho positivo quando realizado sobre o gás:

$$
w_i^{on}
=\frac{P_i v_i}{1-n_i}
\left[1-\left(\frac{v_i}{v_{i+1}}\right)^{n_i-1}\right].
$$

A Primeira Lei é aplicada por unidade de massa,

$$
u_{i+1}=u_i+q_i+w_i^{on},
$$

e o expoente $n_i$ é corrigido iterativamente até que trabalho, energia
interna, temperatura, pressão e equação de estado sejam consistentes. Intervalos
com $v_i\simeq v_{i+1}$ são tratados como isocóricos.

## Validação com os parâmetros do artigo

A implementação foi validada reproduzindo o caso publicado com
\(\delta=10^\circ\). Essa simulação mantém:

| Parâmetro | Valor |
|---|---:|
| Volume deslocado unitário | 250 cm³ |
| Relação biela/manivela, $L/R$ | 5 |
| Taxa de compressão, $r$ | 12 |
| Ângulo de ignição, \(\theta\) | −5° |
| Duração angular da adição de calor, \(\delta\) | 10° |
| Estado inicial | 300 K e 100 kPa |
| Calor específico fornecido, $q_{in}$ | 1.000 kJ/kg |
| Fluido de trabalho | CO₂ |
| Intervalos de compressão e expansão | 90 por processo |
| Passo durante a adição de calor | 0,5° |

O ciclo calculado possui 200 intervalos e 201 estados. A implementação usa o
polinômio de terceiro grau disponível em [`data/data.csv`](data/data.csv) e
obtém os seguintes resultados:

| Indicador | Resultado |
|---|---:|
| Eficiência térmica | 38,409% |
| Trabalho específico de compressão | 197,83 kJ/kg |
| Trabalho específico de expansão | 581,93 kJ/kg |
| Trabalho líquido específico | 384,09 kJ/kg |
| Razão de consumo de trabalho | 0,3400 |
| Pressão máxima | 5.918,86 kPa |
| Temperatura máxima | 1.514,06 K |

A eficiência calculada reproduz os 38,4% publicados. Os cinco diagramas abaixo
são gerados diretamente por [`src/base_case_analysis.py`](src/base_case_analysis.py)
e permanecem versionados em `img/` como evidência reproduzível da validação.

### Diagrama $\log(P)\times\log(v)$

A escala log-log evidencia os subprocessos politrópicos. A linha vertical à
direita é a rejeição isocórica de calor que fecha o ciclo.

![Validação do caso do artigo em diagrama log(P) por log(v)](img/base_case_log_pressure_vs_log_specific_volume.png)

### Diagrama $P\times v$

A área interna representa o trabalho líquido específico de 384,09 kJ/kg. O pico
de pressão ocorre próximo ao PMS, durante a adição finita de calor.

![Validação do caso do artigo em diagrama P por v](img/base_case_pressure_vs_specific_volume.png)

### Pressão por ângulo do virabrequim

A faixa destacada identifica a adição de calor entre −5° e +5°. A pressão máxima
calculada é 5.918,86 kPa.

![Validação da pressão por ângulo do virabrequim](img/base_case_pressure_vs_crank_angle.png)

### Diagrama $T\times v$

A temperatura máxima de 1.514,06 K é atingida ao final da adição de calor. A
expansão converte parte dessa energia interna em trabalho antes da rejeição
isocórica.

![Validação da temperatura por volume específico](img/base_case_temperature_vs_specific_volume.png)

### Expoente politrópico por ângulo do virabrequim

O gráfico mostra a evolução do expoente equivalente em cada intervalo. A escala
simétrica logarítmica preserva a visualização dos valores de grande módulo
próximos ao PMS.

![Validação do expoente politrópico por ângulo do virabrequim](img/base_case_polytropic_exponent_vs_crank_angle.png)

## Interface Python

`simulate_cycle` retorna o histórico termodinâmico completo. Para obter somente
os indicadores de um ponto de operação, use `evaluate_operating_point`. A função
`objective_function` retorna os cinco objetivos segundo uma convenção de
minimização:

1. negativo da eficiência térmica;
2. negativo da potência líquida específica;
3. razão de consumo de trabalho;
4. pressão máxima;
5. temperatura máxima.

```python
from src.FTHA import OBJECTIVE_NAMES, objective_function

objectives = objective_function([4_500.0, -48.0])
print(dict(zip(OBJECTIVE_NAMES, objectives)))
```

## Estrutura do projeto

- `src/FTHA.py`: modelo termodinâmico e função objetivo;
- `src/gas_prop.py`: propriedades do gás ideal com calores específicos
  variáveis;
- `src/base_case_analysis.py`: simulação e diagramas diagnósticos do caso
  \(\delta=10^\circ\);
- `data/data.csv`: coeficientes polinomiais das propriedades dos gases;
- `img/`: artefatos gráficos gerados;
- `tests/`: testes de regressão e da interface.

A localização do CSV e dos diretórios de saída é calculada a partir da raiz do
projeto e não depende do diretório corrente usado para iniciar o Python.

## Ambiente e validação

O projeto usa Python 3.13 e `uv`:

```bash
uv sync
uv run python -m unittest discover -s tests
uv run python -m src.base_case_analysis
uv run python -c "from src.FTHA import objective_function; print(objective_function([4500, -48]))"
```

## Referências

- NAAKTGEBOREN, Christian. An air-standard finite-time heat addition Otto engine
  model. *International Journal of Mechanical Engineering Education*, Londres,
  v. 45, n. 2, p. 103–119, 2017. DOI: 10.1177/0306419016689447.
- ÇENGEL, Y. A.; BOLES, M. A. *Termodinâmica*. 7ª ed. Porto Alegre: Grupo A,
  2013.
