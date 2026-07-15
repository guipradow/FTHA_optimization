# FTHA Optimization

Modelo numérico do ciclo Otto com adição de calor em tempo finito (FTHA,
*Finite-Time Heat Addition*), baseado no trabalho de Naaktgeboren (2017). O
projeto teve origem em um estudo da disciplina de Máquinas Térmicas do curso de
Engenharia Mecânica da UTFPR, campus Guarapuava, e prepara o modelo
termodinâmico para uso posterior em otimização multiobjetivo.

O artigo de referência está em
[`referencias/0306419016689447.pdf`](referencias/0306419016689447.pdf), e o
relatório original está em
[`notebooks/e2.2_case_study.ipynb`](notebooks/e2.2_case_study.ipynb). A
implementação reutilizável encontra-se em [`src/FTHA.py`](src/FTHA.py).

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

## Resultados apresentados no artigo

Esta seção considera somente as cinco figuras publicadas e anexadas ao projeto.
Ela não inclui, por ora, o estudo posterior que combina diferentes rotações e
ângulos de ignição.

### Figura 1 — validação do modelo

A validação compara o ciclo FTHA com a solução analítica do ciclo Otto
ar-padrão usando:

- taxa de compressão $r=8$;
- ar quente com calores específicos constantes e $k=1{,}3343$;
- $T_0=300\ \mathrm{K}$ e $P_0=100\ \mathrm{kPa}$;
- adição de calor quase instantânea, com \(\delta=0{,}01^\circ\), centrada no
  PMS;
- 180 intervalos na compressão e na expansão e dois intervalos na adição de
  calor.

Ambos os modelos fornecem \(\eta_t=50{,}098\%\). No diagrama $P-v$ em escala
log-log, os estados calculados pelo modelo FTHA se sobrepõem às linhas da solução
analítica com pelo menos cinco algarismos significativos. Cada marcador mostra
um estado da discretização politrópica.

[Consultar a Figura 1 no artigo](referencias/0306419016689447.pdf#page=11).

### Figuras 2–5 — efeito da duração angular da adição de calor

A série publicada mantém todos os parâmetros fixos e varia apenas

$$
\delta\in\{10^\circ,30^\circ,50^\circ,70^\circ,90^\circ,110^\circ\}.
$$

Os parâmetros comuns são:

| Parâmetro | Valor |
|---|---:|
| Volume deslocado unitário | 250 cm³ |
| Relação biela/manivela, $L/R$ | 5 |
| Taxa de compressão, $r$ | 12 |
| Ângulo de ignição, \(\theta\) | −5° |
| Estado inicial | 300 K e 100 kPa |
| Calor específico fornecido, $q_{in}$ | 1.000 kJ/kg |
| Fluido de trabalho | CO₂ |
| Intervalos de compressão e expansão | 90 por processo |
| Passo durante a adição de calor | 0,5° |

As eficiências mostradas nas legendas são:

| \(\delta\) | 10° | 30° | 50° | 70° | 90° | 110° |
|---:|---:|---:|---:|---:|---:|---:|
| \(\eta_t\) | 38,4% | 37,0% | 34,1% | 30,6% | 27,0% | 23,7% |

#### Figura 2 — $P-v$ em escala log-log

O gráfico evidencia a discretização dos processos e a mudança da forma do ciclo.
À medida que \(\delta\) aumenta, a eficiência diminui e a pressão ao fim da
expansão cresce, indicando maior potencial de produção de trabalho descartado
com o fluido.

[Consultar a Figura 2 no artigo](referencias/0306419016689447.pdf#page=12).

#### Figura 3 — $P-v$ em escala linear

A área interna do ciclo, proporcional ao trabalho líquido, diminui com o aumento
de \(\delta\). A pressão máxima também cai acentuadamente. Para os dois maiores
valores de \(\delta\), a pressão máxima ocorre ao final da compressão, e não
durante a adição de calor.

[Consultar a Figura 3 no artigo](referencias/0306419016689447.pdf#page=13).

#### Figura 4 — pressão por ângulo do virabrequim

As diferenças entre os ciclos concentram-se ao redor do PMS e no início da
expansão. Uma adição de calor angularmente mais longa reduz e desloca o pico de
pressão, pois uma parcela maior do calor é fornecida enquanto o pistão já se
afasta do PMS.

[Consultar a Figura 4 no artigo](referencias/0306419016689447.pdf#page=13).

#### Figura 5 — temperatura por volume específico

Com o aumento de \(\delta\), o gás leva uma faixa angular maior para aquecer e
atinge a temperatura máxima em volumes específicos progressivamente maiores.
Resta menos expansão após o aquecimento, e o estado no PMI apresenta temperatura
e pressão mais altas. A maior exergia descartada explica a queda monotônica da
eficiência térmica.

[Consultar a Figura 5 no artigo](referencias/0306419016689447.pdf#page=14).

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
- `notebooks/`: versões originais dos estudos;
- `referencias/`: artigo e material bibliográfico;
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
