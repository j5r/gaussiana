import numpy as np
import matplotlib.pyplot as plt

X_GRAFICOS = []
Y_GRAFICOS = []
NOME_GRAFICOS = []


def regressao_polinomial(x, y, grau=1):
    if len(x) != len(y):
        raise Exception('Dimension of x and y does not match.')

    soma_potencias_x = []
    for i in range(2*grau+1):
        soma_potencias_x.append(sum([value**i for value in x]))
    y_vezes_x_n = []
    for i in range(grau+1):
        y_vezes_x_n.append(
            sum([y[index] * x[index]**i for index, _ in enumerate(x)]))
    y_vezes_x_n.reverse()
    y_vezes_x_n = np.asarray(y_vezes_x_n)
    y_vezes_x_n = y_vezes_x_n[:, np.newaxis]

    matriz = np.zeros((grau+1, grau+1))
    index = len(soma_potencias_x)-1
    for eq in range(grau+1):
        for coef in range(grau+1):
            matriz[eq, coef] = soma_potencias_x[index]
            index -= 1
        index += grau
    precond = np.diag(1/np.diag(matriz))

    coeficientes = np.linalg.solve(precond@matriz, precond@y_vezes_x_n)
    def fun(x): return sum([coeficientes[i] * (x ** (grau-i))
                            for i in range(grau+1)])
    return coeficientes, fun


legendas = []


def solve_plot(x, y, grau=1, plot=False):
    coef, fun = regressao_polinomial(x, y, grau)
    if plot:
        plt.plot(x, y, 'k*')
        legendas.append('Dados')
    domain = np.linspace(min(x), max(x), 10*grau)
    image = np.array(list(map(fun, domain)))
    plt.plot(domain, image, '-')
    legendas.append(f'Grau {grau}')
    X_GRAFICOS.append(domain.tolist())
    Y_GRAFICOS.append(image.flatten().tolist())
    NOME_GRAFICOS.append(f'Grau {grau}')
    return coef, fun


global HTML_PAGE
HTML_PAGE = """
<!DOCTYPE html>
<html lang="pt-br">
<header>
  <meta charset="UTF-8">
  <script type="text/x-mathjax-config">
    MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$']]}});
  </script>

  <script type="text/javascript"
    src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
  </script>

  <script src="https://cdn.jsdelivr.net/npm/chart.js@2.9.3/dist/Chart.min.js"></script>

  <style>
  :root{
    font-size: 16px;
    background-color: #f6f6ff;    
    text-justify: newspaper;
    }
    body{
        padding:15px;
    }
    div {
        color: #22f;
        margin-left: auto;
        margin-right: auto;
        text-justify: auto;
        width: 90%;
      }
    div.content{
        margin-top: auto;
    }
    div.equacao{
        padding: 15px; 
        margin-top: 8vh;
        background-color: #e6e6ee;
        border-radius: 15px;
        width: 75vw;
    }    
    canvas{
        width: auto;
        height: auto;   
    }
    div.equacao > span{
        color: red;
    }
    h1,h2{
        margin-left: 10vw;
    }
  </style>
</header>

<body>
<h1> REGRESSÃO LINEAR POLINOMIAL </h1>
<br>
<div style="width:60vw;height:60vh;">
    <canvas id="canvas"  class="chartjs-render-monitor"></canvas>
</div>


<div class="content">
**FUNCOES
</div>

<h2>DADOS</h2>
<div class="equacao" style="margin-top:1vh;">
x = **XX
</div>

<div class="equacao">
y = **YY
</div>

<script>
// =====================================
color_list = ["#d646","#64d6","#4d66","#d466","#46d6","#6d46","","#a886","#8a86"]
XDATA = **XDATA;
YDATA = **YDATA;

NOME_GRAFICOS = **NOME_GRAFICOS;
X_GRAFICOS = **X_GRAFICOS;
Y_GRAFICOS = **Y_GRAFICOS;
// =====================================
function estruturar(x,y){
    if(x.length != y.length){return undefined;}
    result = [];
    for(let i=0; i < x.length; i++){
        result.push({"x":x[i], "y":y[i]});
    }
    return result;
}
// =====================================
ctx = document.getElementById("canvas").getContext("2d");
var options = {
    scales: {
      xAxes: [{
                type: "linear",
                position: "bottom"
            }]
    }
  }
// =====================================
var chartData = {
    datasets: [{
      type: "scatter",
      label: "Dados",
      data: estruturar(XDATA,YDATA),
      showLine: false,
      pointStyle: 'rectRounded',
      borderColor: '#222',
      backgroundColor: '#2228',
    }]
  };
// =====================================

for(let i=0; i < X_GRAFICOS.length; i++){
    var item_nome = NOME_GRAFICOS[i];
    var item_x = X_GRAFICOS[i];
    var item_y = Y_GRAFICOS[i];

    chartData.datasets.push({
      type: 'line',
      label: item_nome,
      data: estruturar(item_x, item_y),
      fill: false,
      pointStyle: 'dash',
      tension: 0,
      borderColor: color_list[i],
    });
};
**INCLUIR_DADOS
// =====================================
window.myChart = Chart.Scatter(ctx, {
    data: chartData,
    options: options
  });
// =====================================

</script>
</body>
</html>
"""


def my_plot(x, y, graus=[0, 1, 2, 3, 4]):
    global HTML_PAGE
    if 0 in graus:
        graus.remove(0)
        HTML_PAGE = HTML_PAGE.replace("**XDATA", repr(x))
        HTML_PAGE = HTML_PAGE.replace("**YDATA", repr(y))
        HTML_PAGE = HTML_PAGE.replace("**INCLUIR_DADOS", "")
    else:
        HTML_PAGE = HTML_PAGE.replace("**XDATA", "[]")
        HTML_PAGE = HTML_PAGE.replace("**YDATA", "[]")
        HTML_PAGE = HTML_PAGE.replace(
            "**INCLUIR_DADOS", "chartData.datasets = chartData.datasets.slice(1);")

    equacoes = []
    for grau in graus:
        coefs, _ = solve_plot(x, y, grau)
        expoente = grau
        equacao = f"$$f_{ {grau} }(x)="
        for coef in coefs:
            coef = str(float(coef))
            decimal = ""
            if coef.find("e") != -1:
                coef, decimal = coef.split("e")
                decimal = f"\\times 10^{ {int(decimal)} }"
            coef = float(coef)
            if coef > 0:
                equacao += "+" + str(coef) + decimal
            else:
                equacao += "-" + str(abs(coef)) + decimal
            if expoente > 0:
                if expoente > 1:
                    equacao += f"x^{ {expoente} } "
                else:
                    equacao += "x "
            expoente -= 1
        equacao = equacao.replace("=+", "=") + "$$"
        equacoes.append('\n<div class="equacao">\n' +
                        '\n<span><a href="https://www.codecogs.com/latex/eqneditor.php" target="_blank">LaTeX: </a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;' +
                        equacao.replace("$", "")+"</span>\n</div>\n")
    equacoes = "".join(equacoes)
    HTML_PAGE = HTML_PAGE.replace('**X_GRAFICOS', repr(X_GRAFICOS))
    HTML_PAGE = HTML_PAGE.replace('**Y_GRAFICOS', repr(Y_GRAFICOS))
    HTML_PAGE = HTML_PAGE.replace('**NOME_GRAFICOS', repr(NOME_GRAFICOS))
    HTML_PAGE = HTML_PAGE.replace("**FUNCOES", equacoes)
    HTML_PAGE = HTML_PAGE.replace("**XX", repr(x))
    HTML_PAGE = HTML_PAGE.replace("**YY", repr(y))

    with open('relatorio.html', 'wb') as f:
        f.write(str.encode(HTML_PAGE))
