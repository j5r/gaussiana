
CONST = 1/Math.sqrt(2*Math.PI)

///////////////
function gauss(z){
    return CONST * Math.exp(-0.5*z*z)
    }

/////////////////
function integral(fun,a,b,dx){
    soma = 0
    upper = Math.max(a,b)
    lower = Math.min(a,b)
    nptos = parseInt( ( upper - lower )/dx + 5)
    dx = (upper - lower)/nptos
    i = 0
    pto = lower
    while(pto<upper){
        pto = lower + i * dx
        soma += dx*fun(pto)
        i++
        }
    return soma
    }

/////////////////////
function normalPadrao(obj){ // {lower:0.3} {upper: 0.5} {dx: 0.001}
    if(obj.l===undefined){lower = -6}
    else{lower = obj.l}

    if(obj.u===undefined){upper = 6}
    else{upper = obj.u}

    if(obj.dx===undefined){dx = 1e-6}
    else{dx = obj.dx}

    // lower upper dx parseFloat(VALUE.toFixed(Math.abs(parseInt(Math.log10(DX)))))
    value = integral(gauss, lower, upper, dx)
    return parseFloat(
            value.toFixed(
                Math.abs(
                    parseInt(
                        Math.log10(dx)
                    )
                )
            )
        )
    }

//////////////////////
function normalInversePadrao(probabilidade,tolerancia=1e-6){
    //calculando intervalo inicial
    bound = {lower: -1, upper: 1}
    while(true){//limitante inferior
        lower = bound.lower
        probInf = normalPadrao({u:lower})
        if(probInf >= probabilidade) {bound.lower-=1}
        else{break}
        }
    while(true){//limitante superior
        upper = bound.upper
        probSup = normalPadrao({u:upper})
        if(probSup <= probabilidade) {bound.upper+=1}
        else{break}
    }
    console.log(bound)


    //calculo via bisecção
    erro = 10
    while(erro>=tolerancia){
        ptoMedio = 0.5*(bound.lower + bound.upper)
        probPtoMedio = normalPadrao({u:ptoMedio,dx:1e-6})
        if(probPtoMedio >= probabilidade){
            bound.upper = ptoMedio
            }
        else if(probPtoMedio <= probabilidade){
            bound.lower = ptoMedio
            }
        erro = Math.abs(bound.upper - bound.lower)
        }
        console.log("Erro menor que a tolerância " +erro)
    return 0.5*(bound.lower + bound.upper)
    }


function normal(obj){ //{l,u,dx}
    //media e desvio padrao
    if(obj.m===undefined){
        Error("Faltando a media 'obj.m'.")
        console.log("Faltando a media 'obj.m'.")
        return undefined
        }
    if(obj.s===undefined && obj.s2===undefined){
        Error("Faltando a variancia 'obj.s' ou o desvio padrao 'obj.s2'.")
        console.log("Faltando a variancia 'obj.s' ou o desvio padrao 'obj.s2'.")
        return undefined
        }
    if(obj.s===undefined){//desvio padrao
        if(obj.s2<0) {
            Error("Variancia precisa ser positiva 'obj.s2'>0.")
            console.log("Variancia precisa ser positiva 'obj.s2'>0.")
            return undefined
        }
        obj.s = Math.sqrt(obj.s2)//variancia
        }
    // limitantes
    if(obj.l===undefined){lower = obj.m - 6*obj.s}
    else{lower = obj.l}

    if(obj.u===undefined){upper = obj.m + 6*obj.s}
    else{upper = obj.u}

    if(obj.dx===undefined){dx = 1e-6}
    else{dx = obj.dx}

    lower = (lower - obj.m)/obj.s
    upper = (upper - obj.m)/obj.s

    return normalPadrao({l:lower,u:upper,dx:dx})
    }




function normalInverse(prob,obj){ //{l,u,dx}
    //media e desvio padrao
    if(obj.m===undefined){
        Error("Faltando a media 'obj.m'.")
        console.log("Faltando a media 'obj.m'.")
        return undefined
        }
    if(obj.s===undefined && obj.s2===undefined){
        Error("Faltando a variancia 'obj.s' ou o desvio padrao 'obj.s2'.")
        console.log("Faltando a variancia 'obj.s' ou o desvio padrao 'obj.s2'.")
        return undefined
        }
    if(obj.s===undefined){//desvio padrao
        if(obj.s2<0) {
            Error("Variancia precisa ser positiva 'obj.s2'>0.")
            console.log("Variancia precisa ser positiva 'obj.s2'>0.")
            return undefined
        }
        obj.s = Math.sqrt(obj.s2)//variancia
        }
    // limitantes



    obj.u = normalInversePadrao(prob)*obj.s + obj.m

    obj.u = parseFloat(
            obj.u.toFixed(
                Math.abs(
                    parseInt(
                        Math.log10(obj.dx/10)
                    )
                )
            )
        )
    obj.l = undefined
    return obj
    }







