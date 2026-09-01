# Relatório de desenvolvimento — hetero-sim

Simulação de uma arquitetura RISC-V heterogênea (CVA6 + Snitch + Spatz) e
classificação de MNIST rodando sobre ela. Este relatório narra o
desenvolvimento em ordem cronológica e reproduz, para cada etapa relevante, a
tabela de desempenho por núcleo medida naquele ponto — para que a evolução das
decisões de projeto fique visível nos números, e não só na descrição.

Todas as tabelas abaixo vêm de execuções reais no simulador GVSoC (não são
estimativas), extraídas do histórico do repositório: das versões sucessivas de
`README.md` e dos JSONs em `results/`.

## 1. Objetivo do projeto

Simular, em GVSoC, um chip heterogêneo com três tipos de núcleo RISC-V:

- **CVA6** — núcleo host 64 bits, orquestrador, com hierarquia de cache
  completa (L1 I$/D$ + L2 + DRAM).
- **Snitch** — núcleo integer pequeno acoplado a um subsistema de ponto
  flutuante desacoplado, com as extensões proprietárias **Xssr** (stream
  semantic registers) e **Xfrep** (floating-point repeat), organizado em
  cluster com scratchpad (TCDM).
- **Spatz** — o mesmo núcleo Snitch, mas com uma unidade vetorial RVV de 4
  lanes anexada, também em cluster.

O fluxo é sempre o mesmo: um operador ONNX (ou uma rede inteira) entra pelo
Deeploy, que gera código C; esse código é compilado para cada núcleo com o
kernel mais adequado ao seu hardware; o binário roda no GVSoC; o pipeline
recolhe ciclos, erro numérico contra a referência ONNX e contadores de cache.

```
ONNX op + inputs  ──Deeploy──►  C code  ──riscv-gcc──►  ELF(s)  ──GVSoC──►  métricas por núcleo
```

O objetivo final — atingido na última etapa — é treinar uma CNN pequena,
exportá-la para ONNX e classificar MNIST rodando a rede inteira sobre o chip
simulado completo (host + os dois clusters), com o Deeploy decidindo
automaticamente em qual núcleo cada nó do grafo roda.

## 2. Linha do tempo

| Data (2026) | Commit(s) | Marco |
|---|---|---|
| 08-03 | `11b5490`…`aabce10` | Infraestrutura base: runtime bare-metal, alvos GVSoC por núcleo, pipeline ONNX→ciclos |
| 08-04–05 | `f2cad44`, `a4516b6` | Memória modelada nos três núcleos; extensões Xssr/Xfrep no Snitch |
| 08-05 | `4265ee1` | **Primeira tabela comparativa por núcleo** |
| 08-05 | `d4aca65` | Plano de arquitetura para a malha heterogênea completa (2.5D/3D) |
| 08-18 | `5963ac8` | Kernels RVV escritos à mão para o Spatz — **tabela vira a favor do Spatz** |
| 08-26 | `c59b127`, `f8350fd`, `8d51ab9` | Placa `hetero_soc` (host + 2 clusters em um único chip), runtime de dispatch, plataforma Deeploy que mapeia cada nó ao núcleo certo |
| 08-27 | `fa45f2c`, `2acdec4` | GEMM real em RVV + 8 núcleos de cômputo por cluster; kernels RVV viram padrão do pipeline — **tabela final de operador único** |
| 08-29 | `1efa098` | **CNN de MNIST classificada na malha completa**, ponta a ponta |
| 09-01 | — | **Keyword spotting com os dois clusters simultâneos**: `hes_post`/`hes_wait`, front-end MFCC, 1,82× sobre o serial |

As seções seguintes detalham cada marco com sua tabela de desempenho.

## 3. Infraestrutura base (03–05/08)

Antes de qualquer comparação fazer sentido, foi preciso: o runtime bare-metal
(crt0, semihosting, scripts de linker) para cada tipo de núcleo; os alvos
GVSoC (`cva6_real`, `snitch_real`, `spatz_real`, e as variantes `_ideal` de
memória de latência zero); o pipeline que leva um operador ONNX até um número
de ciclos por núcleo; e a correção do modelo vetorial do Spatz no GVSoC
(instruções faltantes, uma corrida de *writeback*, uma configuração de vetor
obsoleta lida em execução adiada — ver `README.md`, seção *GVSoC model
fixes*). Nesta fase o sistema de memória ainda não estava modelado (todo
acesso era de um ciclo) e o Snitch ainda não usava suas extensões — por isso
ainda não há uma tabela comparativa significativa.

Em seguida, o sistema de memória passou a ser simulado nos três núcleos (cache
misses, latência de DRAM e banda de *refill* entram na contagem de ciclos), e
o Snitch ganhou seus kernels de ponto flutuante escritos contra Xssr/Xfrep.

## 4. Primeira tabela comparativa (commit `4265ee1`, 05/08)

Um núcleo de cada tipo, memória modelada, Spatz ainda rodando código
autovetorizado pelo compilador (`-O3 -ffast-math`), sem kernel próprio.

| operador | shape | cva6 | snitch | spatz |
|---|---|---|---|---|
| Add | 64 × fp32, elementwise | **863** | 1053 (0.8×) | 1035 (0.8×) |
| MatMul | 2 × (16×32 · 32×8) fp32 | 119.0k | **12.6k (9.4×)** | 15.8k (7.5×) |
| MatMul (custom op) | 32×32×32 fp32 | 477.5k | **43.1k (11.1×)** | 57.6k (8.3×) |
| GEMM | 32×32×32 fp32 + bias | 489.0k | **43.5k (11.2×)** | 60.9k (8.0×) |
| GEMM (int8) | 32×32×32, s8·s8 → s32 | 575.9k | 451.7k (1.3×) | **84.1k (6.8×)** |
| Conv2D + bias | 2×64×32 fp32, 4 filtros 2×8×8, stride 2×4 | 1.86M | **173.9k (10.7×)** | 457.0k (4.1×) |
| Softmax | 512 fp32, 32 linhas de 16 | 36.5k | 51.0k (0.7×) | **32.5k (1.1×)** |

**Leitura:** um único núcleo Snitch vence toda operação de ponto flutuante,
inclusive contra o Spatz de 4 lanes — Xssr remove as cargas e a aritmética de
endereço, Xfrep remove o laço, e o que sobra é a taxa de FMA que a banda do
*stream* permite. O Spatz só ganha em GEMM inteira, onde nem Xssr nem Xfrep
ajudam (ambos operam sobre o *register file* de ponto flutuante) e a
vetorização RVV se aplica diretamente.

## 5. Plano da malha heterogênea completa (commit `d4aca65`, 05/08)

Com o comparativo de núcleo único em mãos, `docs/hetero-mesh-plan.md` traçou o
caminho até um chip único com CVA6 + Snitch/Spatz em memória 2.5D/3D — a
motivação por trás de tudo que vem depois. O documento já registrava as
simplificações deliberadas (2 clusters em vez de 8, clusters homogêneos, D2D
só por DMA) e os riscos assumidos, com destaque para um: *"Phase 2 é o
concentrado de risco — se o dispatch entre clusters não funcionar, as fases
3–5 não têm em que se apoiar."* Esse risco foi endereçado diretamente nas
etapas de 26/08 (seção 7).

## 6. Kernels RVV do Spatz (commit `5963ac8`, 18/08)

Antes desta etapa veio o empacotamento em Docker (`dad3f35`), para rodar o
simulador em qualquer máquina. Em seguida, o Spatz recebeu kernels de
MatMul/GEMM escritos à mão em RVV (via a flag `--spatz-kernels rvv`), em vez
de depender do autovetorizador do GCC.

| operador | shape | cva6 | snitch | spatz |
|---|---|---|---|---|
| Add | 64 × fp32, elementwise | **863** | 1053 (0.8×) | 1035 (0.8×) |
| MatMul | 2 × (16×32 · 32×8) fp32 | 119.0k | 12.6k (9.4×) | **6.8k (17.5×)** |
| MatMul (custom op) | 32×32×32 fp32 | 477.5k | 43.1k (11.1×) | **7.5k (63.3×)** |
| GEMM | 32×32×32 fp32 + bias | 489.0k | 43.5k (11.2×) | **8.4k (58.5×)** |
| GEMM (int8) | 32×32×32, s8·s8 → s32 | 575.9k | 451.7k (1.3×) | **84.1k (6.8×)** |
| Conv2D + bias | 2×64×32 fp32, 4 filtros 2×8×8, stride 2×4 | 1.86M | **173.9k (10.7×)** | 457.0k (4.1×) |
| Softmax | 512 fp32, 32 linhas de 16 | 36.5k | 51.0k (0.7×) | **32.5k (1.1×)** |

**Leitura:** a tabela vira — Spatz passa a vencer toda operação da família
matmul, por 17–63× sobre CVA6 e 2–6× sobre Snitch. A conclusão da etapa
anterior ("Snitch vence tudo em fp32") não era uma verdade de hardware; era um
artefato de o Spatz ainda rodar código de compilador. Conv2D continua com
Snitch, simplesmente porque o Spatz ainda não tinha um kernel próprio para
essa operação — a lacuna mais óbvia a fechar em seguida.

## 7. A SoC heterogênea completa (26–27/08)

Três commits no mesmo dia (`c59b127`, `f8350fd`, `8d51ab9`) transformaram três
placas medidas separadamente em um único chip:

1. **`c59b127`** — a placa `hetero_soc` compõe, num só espaço de endereços e
   numa só simulação: um orquestrador CVA6, um cluster Snitch de 9 núcleos
   (Xssr/Xfrep) e um par Snitch/Spatz de 2 núcleos. Nenhuma placa GVSoC de
   catálogo junta CVA6 e clusters Snitch; `targets/hetero/soc.py` monta isso a
   partir das peças existentes.
2. **`f8350fd`** — runtime de *dispatch*: o host entrega um kernel a qualquer
   um dos dois clusters e lê o resultado de volta (12/12 corretos em MatMul,
   GEMM e Conv2D). O custo em ocioso é zero — os núcleos de cômputo ficam
   parados numa barreira de hardware do modelo, e o núcleo de controle dorme
   em `wfi` até o host escrever no registrador do cluster.
3. **`8d51ab9`** — a plataforma Deeploy hetero: o grafo ONNX inteiro é
   compilado de uma vez, e um modelo de custo escolhe, nó a nó, em qual dos
   três motores (`cva6`, `snitch`, `spatz`) ele roda.

Nesse ponto, o kernel RVV do Spatz ainda era o de núcleo único da seção 6, e o
cluster Spatz ainda tinha só 1 núcleo de cômputo contra os 8 do cluster
Snitch — uma desvantagem de contagem de núcleos alheia ao hardware. A
plataforma recém-criada foi verificada contra as três opções de fixação
(`--pin`) em GEMM/Regular:

| estratégia | ciclos | vs. mapeamento automático |
|---|---|---|
| mapeado automaticamente | **11.647** (→ snitch) | — |
| `--pin snitch` | 11.647 | igual |
| `--pin spatz` | 64.501 | 5,5× mais lento |
| `--pin cva6` | 488.728 | 42× mais lento |

`fa45f2c` corrigiu as duas causas dessa desvantagem no mesmo commit: um
kernel RVV real para MatMul/GEMM que vetoriza as *colunas de saída* em vez do
eixo de redução (B lido em passo unitário, A transmitido por
`vfmacc.vf`, acumulador vetorial vivendo o laço `k` inteiro, sem redução
horizontal), e 8 núcleos de cômputo no cluster Spatz — igualando a contagem
do cluster Snitch. Com os dois clusters em pé de igualdade (`make mesh-test`,
ainda 12/12 corretos):

| kernel | snitch (8 núcleos) | spatz (8 núcleos) | vencedor |
|---|---|---|---|
| MatMul 32×32×32 | 8.048 | 1.952 | spatz, 4,1× |
| GEMM 32×32×32 | 7.258 | 2.752 | spatz, 2,6× |
| Conv2D | 96.568 | 86.285 | spatz, 1,1× (ainda autovetorizado no Spatz) |

E o mapeamento automático em GEMM/Regular se inverteu:

| estratégia | ciclos | vs. mapeamento automático |
|---|---|---|
| mapeado automaticamente | **7.885** (→ spatz) | — |
| `--pin spatz` | 7.885 | igual |
| `--pin snitch` | 11.647 | 1,5× mais lento |
| `--pin cva6` | 488.728 | 62× mais lento |

`2acdec4` fechou a etapa tornando os kernels RVV o padrão do pipeline de
núcleo único (não mais uma flag opcional), com a tabela final da família de
operadores isolados:

| operador | shape | cva6 | snitch | spatz |
|---|---|---|---|---|
| Add | 64 × fp32, elementwise | **863** | 1053 (0.8×) | 1035 (0.8×) |
| MatMul | 2 × (16×32 · 32×8) fp32 | 119.0k | 12.6k (9.4×) | **7.0k (16.9×)** |
| MatMul (custom op) | 32×32×32 fp32 | 477.5k | 43.1k (11.1×) | **11.2k (42.6×)** |
| GEMM | 32×32×32 fp32 + bias | 489.0k | 43.5k (11.2×) | **13.5k (36.3×)** |
| GEMM (int8) | 32×32×32, s8·s8 → s32 | 575.9k | 451.7k (1.3×) | **84.1k (6.8×)** |
| Conv2D + bias | 2×64×32 fp32, 4 filtros 2×8×8, stride 2×4 | 1.86M | **173.9k (10.7×)** | 457.0k (4.1×) |
| Softmax | 512 fp32, 32 linhas de 16 | 36.5k | 51.0k (0.7×) | **32.5k (1.1×)** |

**Leitura:** a linha divisória entre os dois núcleos acelerados não é
ponto-flutuante versus inteiro — é se o trabalho se reduz a um laço denso de
multiplicação-acumulação que o sequenciador do Snitch consegue reproduzir. Se
sim, Snitch é competitivo; se não (uma redução inteira, uma chamada de libm),
o Spatz é o cluster melhor, qualquer que seja o tipo de dado. Conv2D
permanece com o Snitch por lacuna de software (falta um kernel RVV
escrito à mão para o Spatz), não por limite de hardware.

## 8. MNIST na malha completa (commit `1efa098`, 29/08)

`pipeline/mnist.py` treina uma CNN pequena em MNIST usando apenas numpy (sem
nova dependência), exporta para ONNX e a empacota como operador do pipeline. A
rede:

```
28×28×1 → Conv 3×3,8 → 26×26×8 → ReLU → MaxPool 2 → 13×13×8
        → Conv 3×3,16 → 11×11×16 → ReLU → MaxPool 2 → 5×5×16
        → Flatten → 400 → Gemm → 32 → ReLU → Gemm → 10 → Softmax
```

Duas verificações independentes: a passada numpy bate com o onnxruntime sobre
o grafo exportado (diferença máxima de 2,0×10⁻⁷), e a acurácia relatada
confirma que o treino funcionou — **97,61% sobre as 10.000 imagens de
teste**, 4 épocas. `runtime/mesh/mnist_main.c` roda cada imagem embutida na
malha completa e checa a predição contra o rótulo verdadeiro *e* contra o
onnxruntime sobre o mesmo grafo — a segunda é a que pode falhar a simulação:
um dígito errado só diz que a rede vale pouco; uma discordância do
onnxruntime diria que o chip simulado computou algo diferente da referência.

Execução completa (64 imagens): **100% de acordo com o onnxruntime**, 100% de
acerto nas 64 imagens, **570.029 ciclos por imagem**.

Comparação de posicionamento (8 imagens, mesma metodologia `--pin` das seções
anteriores, agora sobre a rede inteira):

| estratégia | ciclos/imagem | vs. mapeamento automático |
|---|---|---|
| mapeado automaticamente | **575.872** | — |
| `--pin spatz` | 575.872 | igual — o mapeador já escolhe Spatz onde pode |
| `--pin snitch` | 621.111 | 1,08× mais lento |
| `--pin cva6` | 4.215.076 | 7,3× mais lento |

O log da execução automática confirma nó a nó o que a seção 7 já indicava:
`Conv` e `Gemm` (as operações densas) vão para `spatz`; `Relu`, `MaxPool`,
`Reshape` e `Softmax` ficam no `cva6`, coerente com nenhum dos clusters ter
kernel para elas. Ciclos por engine, na execução de 8 imagens: **spatz
2.418.667**, **cva6 2.140.367** — as duas cargas ficam próximas porque o
CVA6 absorve todo o trabalho de controle e as operações *softmax*/ativação,
não porque seja competitivo nas mesmas operações densas do Spatz.

Este marco fecha o objetivo original do projeto: uma rede neural real,
treinada e verificada contra uma referência externa, classificando dados
reais rodando ponta a ponta sobre a arquitetura heterogênea simulada — com o
compilador escolhendo automaticamente, por nó do grafo, o núcleo certo para
cada operação.

## 9. Os dois clusters ao mesmo tempo: keyword spotting

A seção 8 fecha o objetivo original, mas deixa duas coisas visíveis na própria
tabela. Na malha completa rodando MNIST, o cluster **Snitch executa zero
ciclos** — o modelo de custo manda, corretamente, todo nó denso para o Spatz — e
`hes_offload()` é bloqueante, de modo que mesmo quando dois clusters poderiam
trabalhar juntos eles se revezam: o tempo total é a *soma* dos motores, nunca o
*máximo*. O que a seção 8 mede é escolha de operador, não computação
heterogênea.

`ops/kws` é uma aplicação construída para precisar dos dois. É *keyword
spotting* sobre fala sintética, e tem dois estágios que querem hardware
diferente e estão disponíveis ao mesmo tempo:

```
 clipe N+1 ─┐
            ▼
  [cluster SNITCH]  janela → FFT → |·|² → mel → log → DCT     um job, 8 núcleos,
            │       (HES_K_MFCC_FP32)                         fatiado por quadro
            ▼  atributos 32×13   (memória principal, buffer duplo)
  [cluster SPATZ]   Conv 3×3 → Conv 3×3 → Gemm → Gemm         gerado pelo Deeploy,
            │                                                 kernels existentes
            ▼  logits
  [CVA6]            ReLU / MaxPool / Softmax / argmax / controle
```

Como os clipes chegam continuamente, o *front-end* do clipe N+1 é independente
do classificador do clipe N. `runtime/mesh/kws_main.c` posta um antes de rodar o
outro e o recolhe depois. `RunNetwork()` despacha seus nós Conv e Gemm para o
mailbox de *outro* cluster, então a sobreposição não exigiu nada do código
gerado — apenas dividir `hes_offload()` em `hes_post()` + `hes_wait()`. Cada
cluster já tinha seu próprio par `seq`/`done_seq`: um job por cluster em voo
sempre foi expressável, `hes_offload()` é que nunca usou isso.

### 9.1 O que a sobreposição vale

16 clipes, memória modelada, `make kws`:

| estratégia | ciclos/clipe | vs. pipelined |
|---|---|---|
| front-end no snitch, classificador no spatz | **256.652** | — |
| o mesmo trabalho, serial (`SERIAL=1`) | 468.232 | 1,82× mais lento |
| front-end no spatz, classificador no snitch (`FE=spatz PIN=snitch`) | 269.434 | 1,05× mais lento |
| classificador no host (`PIN=cva6`, 8 clipes) | 1.539.740 | 6,0× mais lento |

E, pela primeira vez neste repositório, os três motores fazem trabalho real:

| motor | ciclos (16 clipes) | fração | |
|---|---|---|---|
| snitch | 3.615.971 | 50,2% | o front-end MFCC |
| spatz | 1.831.164 | 25,4% | Conv e Gemm |
| cva6 | 1.755.342 | 24,4% | ReLU, MaxPool, Softmax, controle |

contra o `spatz 53% / cva6 47% / snitch 0%` do MNIST. **93,4% dos ciclos do
front-end se sobrepõem ao classificador** e não custam tempo de parede algum.

### 9.2 O que a comparação de posicionamento diz

As duas linhas do meio da primeira tabela são o mesmo trabalho com os estágios
trocados entre os clusters, e o interessante é que o Spatz é mais rápido nos
**dois** estágios e ainda assim perde:

| estágio | no snitch | no spatz | |
|---|---|---|---|
| front-end MFCC | 226,0k/clipe | **185,7k/clipe** | spatz, 1,22× |
| classificador (parte do cluster) | 129,8k/clipe | **114,4k/clipe** | spatz, 1,13× |

Os dois estágios *precisam* estar em clusters diferentes: o mailbox de um
cluster guarda um único descritor, então postar o próximo front-end no cluster
que o classificador está usando sobrescreveria um job em voo — o que `hes_post()`
recusa e `pipeline/run_hetero.py` detecta antes da compilação. A escolha,
portanto, é qual estágio recebe o Spatz. E o período de um *pipeline* é o
**máximo** dos seus estágios, não a soma: sob qualquer das duas atribuições o
classificador é o estágio mais lento, então o Spatz pertence ao classificador e
o front-end fica com o Snitch por eliminação.

A regra não é "cada estágio no núcleo que o roda mais rápido"; é **"o núcleo
melhor para o estágio que determina o período"**. Essa é a conclusão que a
seção 8 não tinha como produzir, porque nada lá rodava em paralelo.

### 9.3 O que o kernel SSR do front-end compra — e o que não compra

`runtime/snitch/kernels/mfcc_fp32_ssr.c` transmite dois dos quatro estágios, os
dois em que o argumento de padrão de acesso se apoia: o **banco de filtros mel**
(40 reduções, cada uma sobre sua própria faixa irregular de 10–30 bins — vetores
curtos *e* uma redução horizontal por filtro, o pior caso do RVV) e a **DCT-II**.

Vale 8,2% do front-end (992,5k → 911,6k ciclos em 4 clipes), e nada além disso,
porque a **FFT domina e não é transmitida**: sua borboleta quer quatro leituras
e quatro escritas por iteração contra três *data movers*, então transmiti-la
significa quebrá-la em passagens por um buffer intermediário, e se isso custa
menos do que as cargas que remove é uma pergunta real, não retórica. Fica em
aberto.

É também por isso que o Spatz vence o front-end na tabela acima: **a medição não
sustenta a hipótese a priori** de que o front-end seria trabalho de formato
Snitch. Isso é registrado como refutado, e não silenciosamente abandonado — na
mesma disciplina da seção 6, onde a conclusão "Snitch vence tudo em fp32" também
não sobreviveu à medição seguinte.

### 9.4 Como é verificado

Três checagens independentes, e errar a palavra-chave não é nenhuma delas:

- a predição de cada clipe contra o **onnxruntime** sobre o mesmo grafo — 16/16,
  a mesma disciplina de `mnist_main.c`;
- os **atributos** de cada clipe contra o front-end numpy de `pipeline/kws.py`,
  embutidos em `ops/kws/kws_data.h`. Máx |chip − numpy| = 5×10⁻⁶. Sem isso, uma
  FFT errada só apareceria como uma classificação errada;
- `make ssr-test` roda `runtime/tests/ssr_mfcc.c`, que compara o banco de
  filtros e a DCT transmitidos contra uma referência escalar nas larguras de
  filtro irregulares e contagens de cepstra que a aplicação nunca alcança —
  idêntico bit a bit nos cinco casos.

O áudio é sintetizado proceduralmente a partir de moldes de formantes (sem
download, sem dependência nova), então os 98,8% de acurácia de teste dizem que a
rede treinou, não que o modelo compete com a literatura de Speech Commands. O
que está sendo medido é o chip.

## 10. Resultado final consolidado

As duas tabelas abaixo, lado a lado, resumem a evolução completa do projeto:
operador isolado (núcleo único, seção 7) e rede completa (malha inteira,
seção 8).

| | escala | melhor núcleo por operação densa | ganho vs. CVA6 |
|---|---|---|---|
| Operador isolado (`2acdec4`) | 1 núcleo por tipo | Spatz (matmul/GEMM), Snitch (Conv2D) | 17–63× |
| MNIST na SoC completa (`1efa098`) | 8+8 núcleos de cômputo por cluster, rede inteira | Spatz (Conv, Gemm); CVA6 (Relu/Pool/Reshape/Softmax) | 7,3× ponta a ponta |
| KWS nos dois clusters | front-end e classificador simultâneos | Snitch (front-end MFCC), Spatz (Conv/Gemm), CVA6 (ativações) | 1,82× sobre o mesmo trabalho serial |

A trajetória em três frases: **Snitch domina** enquanto o Spatz roda código
de compilador (seção 4); **Spatz domina** assim que ganha um kernel RVV
escrito à mão (seções 6–7); e, na malha completa rodando uma CNN real, o
**mapeamento automático do Deeploy reproduz exatamente** essa mesma decisão
por operação — sem que fosse necessário fixar nada à mão (seção 8).

## 11. Limitações e trabalhos futuros

- **Conv2D no Spatz ainda é autovetorizado**, não um kernel RVV escrito à
  mão — é a lacuna mais clara deixada em aberto pela seção 7; o ganho de
  1,1× sobre o Snitch em 8 núcleos vem só da contagem de núcleos, não de um
  kernel melhor.
- **Kernel int8 genérico em ambos os clusters** — nem Snitch nem Spatz têm
  kernel escrito à mão para GEMM inteira; 84,1k ciclos (seção 4) não é um
  teto.
- **Redes inteiras rodam só em fp32** — a plataforma Deeploy hetero recusa
  qualquer rede que não seja fp32 nos clusters, porque o tipo de dado não é
  lido diretamente dos dtypes do ONNX (`8d51ab9`); redes inteiras hoje rodam
  inteiramente no host.
- **Escala de clusters** — o plano original (`docs/hetero-mesh-plan.md`)
  previa avaliar 8 clusters de 8 núcleos; o projeto ficou em 2 clusters de 8
  núcleos de cômputo cada, por custo de tempo de simulação (73 instâncias de
  ISS versus as atuais ~18). Fica como pergunta em aberto do plano original:
  publicar os resultados finais em 2×8 ou justificar a extrapolação para
  8×8.
- **D2D e HyperRAM (memória 2.5D/3D)** do plano original não foram
  implementados — o projeto convergiu para DRAM única antes de chegar a essa
  fase.
- **A FFT do front-end KWS não é transmitida por SSR** (seção 9.3). É a lacuna
  mais clara deixada pela seção 9, e a razão pela qual o Spatz vence um estágio
  que o argumento de padrão de acesso previa para o Snitch.
- **Não existe kernel RVV escrito à mão para o front-end no Spatz**, assim como
  não existe para Conv2D: os 185,7k ciclos/clipe do Spatz são código
  autovetorizado, não um teto.
- **Não há caminho cluster a cluster**: os atributos vão do Snitch para a
  memória principal e de lá para o Spatz. É um ida-e-volta de DMA real, contado
  honestamente nos números acima, e é também por que o front-end é um único job
  em vez de três.
- **Um job por cluster em voo.** O mailbox guarda um descritor, o que basta para
  o pipeline de dois estágios desta seção mas impede, por exemplo, dois clipes
  em voo no mesmo cluster.
