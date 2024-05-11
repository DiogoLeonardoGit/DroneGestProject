O script possui um mecanismo de criação de backups automaticos por cada vez que é executado. Lida também, automaticamente, com algumas situações de erro documentadas. 

- Por vezes o sensor deixa de responder sem razão aparente e deixa de retornar valores nas leituras, quando isto acontece devemos terminar de imediato a execução do script. O script está preparado para lidar com situações de termino abrupto pelo que não haverá problema.

- Por vezes existe um delay no sensor que faz com que a leitura seja capturada antes do tempo e provoca um out of range error no array (list index out of range), de igual modo, basta re-executar o script e continuar a gravação de movimentos, o script lida automaticamente com esta situação de modo a que o dataset não seja corrumpido.