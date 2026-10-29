"""Entrena el intérprete estadístico (motor "bayes" del backend): Naive Bayes multinomial.

- Rasgos: palabras sin tildes y pares de palabras seguidas (backend/intent_bayes.rasgos, el mismo código).
- Suavizado de Laplace (alfa): cuánto se le suma a cada conteo para que una palabra que no apareció con una
  acción no tenga probabilidad cero. Se elige con validación.
- Calibración: Naive Bayes da probabilidades demasiado extremas. Se ajusta una temperatura (un solo número que
  divide los logaritmos antes del softmax) minimizando la log-verosimilitud en validación; no cambia qué acción
  gana, solo cuánta confianza se informa. Se reporta el error de calibración (ECE) antes y después.
Validación: las frases a mano de eval/frases_validacion.csv, frases_objetos_validacion.csv y
frases_comandos_validacion.csv (la prueba se mide con eval/eval_intent.py --motor bayes).

Uso (desde la raíz):  backend/.venv/Scripts/python ml/intencion/entrenar_bayes.py
Salida: backend/modelos/bayes.json y ml/resultados/bayes_reporte.json
"""
import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

sys.stdout.reconfigure(encoding="utf-8")
AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ / "backend"))
sys.path.insert(0, str(RAIZ / "eval"))
import intent_bayes  # noqa: E402
from estadistica import ajustar_temperatura, con_temperatura, ece, intervalo_bootstrap  # noqa: E402
from intent_modelo import empaquetar  # noqa: E402

SALIDA = RAIZ / "backend" / "modelos" / "bayes.json"
REPORTE = RAIZ / "ml" / "resultados" / "bayes_reporte.json"
VALIDACION = ["frases_validacion.csv", "frases_objetos_validacion.csv", "frases_comandos_validacion.csv"]
ALFAS = [0.01, 0.03, 0.1, 0.3, 1.0]
MINIMO = 2   # un rasgo tiene que aparecer al menos 2 veces para entrar (los demás no generalizan)


def leer(ruta):
    with open(ruta, encoding="utf-8") as f:
        return [(fila["frase"], fila["accion"]) for fila in csv.DictReader(f)]


def main():
    entrenamiento = leer(AQUI / "frases_entrenamiento.csv")
    validacion = [x for nombre in VALIDACION for x in leer(RAIZ / "eval" / nombre)]
    vec = CountVectorizer(analyzer=intent_bayes.rasgos, min_df=MINIMO)
    X = vec.fit_transform([f for f, _ in entrenamiento])
    y = [a for _, a in entrenamiento]
    Xv = vec.transform([f for f, _ in validacion])
    yv = [a for _, a in validacion]
    print(f"{len(entrenamiento)} frases · {X.shape[1]} rasgos · validación: {len(validacion)} frases a mano")

    resultados = []
    for alfa in ALFAS:
        nb = MultinomialNB(alpha=alfa).fit(X, y)
        exactitud = float((nb.predict(Xv) == np.array(yv)).mean())
        resultados.append({"alfa": alfa, "exactitud_validacion": round(exactitud, 4)})
        print(f"  alfa {alfa:<5} exactitud en validación {exactitud:.3f}")
    alfa = max(resultados, key=lambda r: (r["exactitud_validacion"], -r["alfa"]))["alfa"]
    nb = MultinomialNB(alpha=alfa).fit(X, y)
    clases = list(nb.classes_)

    # Calibración: log P(acción | frase) sin normalizar, igual que en el backend
    log_post = (Xv @ nb.feature_log_prob_.T) + nb.class_log_prior_
    etiquetas = np.array([clases.index(a) for a in yv])
    T = ajustar_temperatura(log_post, etiquetas)
    antes, despues = con_temperatura(log_post, 1.0), con_temperatura(log_post, T)
    acierto = antes.argmax(1) == etiquetas
    calibracion = {"temperatura": round(T, 3), "ece_antes": round(ece(antes.max(1), acierto), 4),
                   "ece_despues": round(ece(despues.max(1), acierto), 4),
                   "confianza_media_antes": round(float(antes.max(1).mean()), 3),
                   "confianza_media_despues": round(float(despues.max(1).mean()), 3),
                   "exactitud": round(float(acierto.mean()), 3)}
    print(f"alfa elegido {alfa} · temperatura {T:.2f}: ECE {calibracion['ece_antes']:.3f} → {calibracion['ece_despues']:.3f}"
          f" (confianza media {calibracion['confianza_media_antes']} → {calibracion['confianza_media_despues']},"
          f" exactitud {calibracion['exactitud']})")

    datos = {"tipo": "naive_bayes_multinomial", "acciones": clases, "alfa": alfa, "temperatura": round(T, 4),
             "rasgos": list(vec.get_feature_names_out()), "log_previa": nb.class_log_prior_.round(5).tolist(),
             "log_verosimilitud": empaquetar(nb.feature_log_prob_.T)}
    SALIDA.write_text(json.dumps(datos, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # Control: el backend con estos pesos da lo mismo que scikit-learn (float16 alcanza)
    modelo = intent_bayes.NaiveBayes(json.loads(SALIDA.read_text(encoding="utf-8")))
    iguales = np.mean([modelo.acciones[int(np.argmax(modelo.probabilidades(f)))] == p
                       for (f, _), p in zip(validacion, nb.predict(Xv))])
    print(f"{SALIDA.relative_to(RAIZ)}: {SALIDA.stat().st_size / 1e6:.1f} MB · coincide con scikit-learn en {iguales:.1%}")

    REPORTE.write_text(json.dumps({
        "fecha": str(date.today()), "frases_entrenamiento": len(entrenamiento), "rasgos": X.shape[1],
        "alfas": resultados, "alfa_elegido": alfa, "calibracion": calibracion,
        "exactitud_validacion_ic95": [round(v, 3) for v in intervalo_bootstrap(acierto)],
        "coincidencia_backend": round(float(iguales), 4),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"reporte -> {REPORTE.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
