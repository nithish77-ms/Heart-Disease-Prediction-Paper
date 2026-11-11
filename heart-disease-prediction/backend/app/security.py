import os, hashlib, hmac, secrets, json, base64, time
SECRET_KEY=os.getenv('JWT_SECRET_KEY','CHANGE_ME_BEFORE_DEPLOYMENT_TO_A_LONG_RANDOM_SECRET')
TOKEN_MINUTES=int(os.getenv('TOKEN_MINUTES','720'))
def hash_password(password: str) -> str:
    salt=secrets.token_bytes(16)
    digest=hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return 'scrypt$'+salt.hex()+'$'+digest.hex()
def verify_password(password: str, stored: str) -> bool:
    try:
        kind,salt_hex,digest_hex=stored.split('$')
        if kind!='scrypt': return False
        actual=hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1)
        return hmac.compare_digest(actual.hex(), digest_hex)
    except Exception: return False
def _b64(data: bytes) -> str: return base64.urlsafe_b64encode(data).rstrip(b'=').decode()
def _unb64(value: str) -> bytes: return base64.urlsafe_b64decode(value + '=' * (-len(value) % 4))
def create_token(user_id: int, role: str):
    if SECRET_KEY.startswith('CHANGE_ME_') and os.getenv('APP_ENV')=='production':
        raise RuntimeError('Set JWT_SECRET_KEY before production deployment.')
    header=_b64(json.dumps({'alg':'HS256','typ':'JWT'},separators=(',',':')).encode())
    payload=_b64(json.dumps({'sub':str(user_id),'role':role,'exp':int(time.time())+TOKEN_MINUTES*60},separators=(',',':')).encode())
    body=header+'.'+payload
    signature=_b64(hmac.new(SECRET_KEY.encode(),body.encode(),hashlib.sha256).digest())
    return body+'.'+signature
def decode_token(token: str):
    try:
        header,payload,signature=token.split('.')
        body=header+'.'+payload
        expected=_b64(hmac.new(SECRET_KEY.encode(),body.encode(),hashlib.sha256).digest())
        if not hmac.compare_digest(signature,expected): raise ValueError('Invalid signature')
        h=json.loads(_unb64(header)); data=json.loads(_unb64(payload))
        if h.get('alg')!='HS256' or int(data.get('exp',0))<int(time.time()): raise ValueError('Expired token')
        return data
    except Exception as e: raise ValueError('Invalid or expired token') from e
