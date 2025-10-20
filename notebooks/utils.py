# utils_soap.py
from zeep import Client

def safe_serialize(obj):
    """
    Serializa objetos Zeep o SOAP a estructuras Python básicas (dict, list, str, etc.)
    para poder normalizarlos con pandas.json_normalize sin errores.
    """
    # tipos base
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [safe_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}

    # intentar con zeep.helpers.serialize_object
    try:
        from zeep.helpers import serialize_object
        ser = serialize_object(obj)
        if ser is not None:
            return safe_serialize(ser)  # recursivo por si hay anidaciones
    except Exception:
        pass

    # objetos Zeep suelen tener __values__
    vals = getattr(obj, "__values__", None)
    if isinstance(vals, dict):
        return {k: safe_serialize(v) for k, v in vals.items()}

    # fallback genérico
    try:
        return {k: safe_serialize(v) for k, v in vars(obj).items()}
    except Exception:
        return obj


def listar_ops(wsdl_url):
    """
    Lista las operaciones disponibles en un servicio SOAP dado su WSDL.
    Útil para explorar endpoints de la Cámara de Diputados o del Senado.
    """
    c = Client(wsdl=wsdl_url)
    for svc in c.wsdl.services.values():
        for port in svc.ports.values():
            binding = port.binding
            for name, op in binding._operations.items():
                args = []
                if op.input and op.input.body and op.input.body.type:
                    args = [elt[0] for elt in op.input.body.type.elements]  # nombres de parámetros
                print(f"{name}({', '.join(args)})")
