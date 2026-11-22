# Frames de referência

Os frames que a suíte de testes usa como gabarito. Ficam **aqui e não em
`frames/`** por dois motivos:

1. `frames/` é gitignored — testes que dependem dele passam a **pular em
   silêncio** num clone novo, o que é pior que falhar.
2. `frames/` é rotacionado durante uma run (ADR-067). Um frame de gabarito
   apagado no meio de uma sessão longa levaria à mesma falha silenciosa.

Cada arquivo aqui foi conferido olhando a imagem, e o que se espera dele está
escrito no teste que o usa. Não substitua nenhum sem reconferir o gabarito.
