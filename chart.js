let FORMDATA = {calcprob: 1, dx: 0.00001}

let warnColor = "#B22424"
let inputColor = "#8EB240"
let disabledColor = "#8A996B"
let enabledColor = "#C8E68A"
let buttonColor = "#fC8A5E"

function getMedia(){
    media = document.getElementById("input-media")
    media.value = media.value.replace(",",".")

    if(isNaN(parseFloat(media.value)) && media.value!==""){
        media.style.backgroundColor = warnColor
        }
    else{
        media.style.backgroundColor = inputColor
        FORMDATA.m = parseFloat(media.value)
        console.log(FORMDATA)
        }

    }

function getVariancia(){
    variancia = document.getElementById("input-variancia")
    variancia.value = variancia.value.replace(",",".")
    valor = variancia.value
    if(isNaN(parseFloat(variancia.value)) || valor <=0 && variancia.value!==""){
        variancia.style.backgroundColor = warnColor
        }
    else{
        variancia.style.backgroundColor = inputColor
        FORMDATA.s2 = parseFloat(variancia.value)
        console.log(FORMDATA)
        }

    }

function getX1(){
    x1 = document.getElementById("input-x1")
    x1.value = x1.value.replace(",",".")

    if(isNaN(parseFloat(x1.value)) && x1.value!==""){
        x1.style.backgroundColor = warnColor
        return 0
        }
    else{
        x1.style.backgroundColor = inputColor
        FORMDATA.l = parseFloat(x1.value)
        FORMDATA.calcprob = 1
        console.log(FORMDATA)
        if(FORMDATA.l >= FORMDATA.u) {
            document.getElementById("input-x1").style.backgroundColor = warnColor
            document.getElementById("input-x2").style.backgroundColor = warnColor
        }
        else{
            document.getElementById("input-x1").style.backgroundColor = inputColor
            document.getElementById("input-x2").style.backgroundColor = inputColor
        }
        return 1
        }
    }

function getX2(){
    x2 = document.getElementById("input-x2")
    x2.value = x2.value.replace(",",".")

     if(isNaN(parseFloat(x2.value)) && x2.value!==""){
        x2.style.backgroundColor = warnColor
        return 0
        }
    else{
        x2.style.backgroundColor = inputColor
        FORMDATA.u = parseFloat(x2.value)
        FORMDATA.calcprob = 1

        xis = document.getElementById("x")
        xis.style.backgroundColor = enabledColor
        prob = document.getElementById("probabilidade")
        prob.style.backgroundColor = disabledColor
        console.log(FORMDATA)
        if(FORMDATA.l >= FORMDATA.u) {
            document.getElementById("input-x1").style.backgroundColor = warnColor
            document.getElementById("input-x2").style.backgroundColor = warnColor
        }
        else{
            document.getElementById("input-x1").style.backgroundColor = inputColor
            document.getElementById("input-x2").style.backgroundColor = inputColor
        }
        return 1
        }
    }

function getProbabilidade(){
    probabilidade = document.getElementById("input-probabilidade")
    probabilidade.value = probabilidade.value.replace(",",".")

    valor = probabilidade.value
     if(isNaN(parseFloat(valor)) || valor <=0 || valor >=1
                || valor.search(",")>=0  && valor!==""){
        probabilidade.style.backgroundColor = warnColor
        return 0
        }
    else{
        probabilidade.style.backgroundColor = inputColor
        FORMDATA.prob = parseFloat(probabilidade.value)
        FORMDATA.calcprob = 0

        xis = document.getElementById("x")
        xis.style.backgroundColor = disabledColor
        prob = document.getElementById("probabilidade")
        prob.style.backgroundColor = enabledColor
        console.log(FORMDATA)
        return 1
        }
    }

function setX(valorx2){
    x2 = document.getElementById("input-x2")
    x2.value = valorx2
    }

function setProbabilidade(valor){
    prob = document.getElementById("input-probabilidade")
    prob.value = valor
    }

function erroDeCalculo(){
    botao = document.getElementById("calcular")
    botao.style.backgroundColor = warnColor
    }

function okCalculo(){
    botao = document.getElementById("calcular")
    botao.style.backgroundColor = buttonColor
    }

function calcular(){
    document.getElementById("calcular").innerText = "Aguarde"
    if(getMedia() || getVariancia()){erroDeCalculo()}else{okCalculo()}

    if(FORMDATA.calcprob){
        //calcula probabilidade dados x1 e x2
        document.getElementById("input-x1").style.display = "block"
        document.getElementById("span-le").style.display = "block"
        console.log(FORMDATA)
        FORMDATA.prob = normal(FORMDATA)
        setProbabilidade(FORMDATA.prob)
        console.log(FORMDATA.prob)

        }
    else{
        //calcula x2 dada a probabilidade
        document.getElementById("input-x1").style.transition = "all 0.9s"
        document.getElementById("input-x1").style.display = "none"
        document.getElementById("span-le").style.transition = "all 0.9s"
        document.getElementById("span-le").style.display = "none"

        FORMDATA = normalInverse(FORMDATA.prob,FORMDATA)
        setX(FORMDATA.u)
        console.log(FORMDATA.u)

        }

        document.getElementById("calcular").setAttribute("onclick","location.reload()")
        document.getElementById("calcular").innerText = "Restart"
        chart(FORMDATA)
    }




function linspace(a,b,n){
    dx = (b-a)/n
    v = []
    for(i=0;i<=n;i++){
        v.push(a+i*dx)
        }
    return v
    }
function image(f,d){
    v = []
    for(i of d){
        v.push(f(i))
        }
    return v
    }
function chart(obj){
    obj.s = Math.sqrt(obj.s2)
    lower = obj.m - 3*obj.s
    upper = obj.m + 3*obj.s
    function aux(x){
        return gauss((x-obj.m)/obj.s)
        }
    domain = linspace(lower,upper,300)
    imagem = image(aux,domain)

    DADOS_FUNCAO = estruturar(domain,imagem)//
    CHART.chart.config.data.datasets[0].data = DADOS_FUNCAO


        if(obj.u===undefined){
            obj.u = upper
            }
        if(obj.l===undefined){
            obj.l = lower
            }

    domain = linspace(obj.l,obj.u,300)


    imagem = image(aux,domain)

    INTEGRAL = estruturar(domain,imagem)
    CHART.chart.config.data.datasets[1].data = INTEGRAL
    CHART.chart.config.data.datasets[1].label= "Probabilidade = "+obj.prob
    CHART.update();
    }





function estruturar(v1,v2){if(v1.length!=v2.length){throw Error("Dados dos eixos X e Y precisam ser em mesma quantidade.");return undefined}if(v1.length==0){throw Error("Não há dados para mostrar.");return undefined}ans = [];for(i=0;i<v1.length;i++){ans.push({"x":v1[i],"y":v2[i]})}return ans}
function gerarDados(a,b,dx=0.1){X=[a];while(true){valor=X[X.length-1]+dx;X.push(valor);if(valor>b){X.pop();break;}}Y = X.map(gauss);return estruturar(X,Y);}

DADOS = gerarDados(-5,5)

var CHART = new Chart(document.getElementById("canvas"),
    {
    type: 'line',
    data: {
    datasets: [{
        label: 'Distribuição Normal',
        data: DADOS,
        fill:false,
        borderColor:"rgb(75, 192, 192)",
        lineTension:0.1
        },
        {
        label: "Probabilidade = ?",
        data: DADOS.slice(0,225),
        fill:true,
        borderColor:"rgb(75, 192, 192)",
        lineTension:0.1
        }]
    },
    options: {
        scales: {
            xAxes: [{
                type: 'linear',
                position: 'bottom'
                }]
            }
        }
    }
)


