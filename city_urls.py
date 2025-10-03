"""
City Data URLs Configuration

This file contains the download URLs for geospatial data for multiple Mexican cities.
Each city requires 6 data files: boundary, pedestrian_network, population_count, 
poi_food, grid, and centroids.
"""

CITY_URLS = {
    "Ciudad de Mexico": {
        "boundary": "https://www.dropbox.com/scl/fi/ut8v6r54bf7votisjsinp/ZM_CDMX.gpkg?rlkey=cuvgo96rcddwgwbiwcqtbhgh8&st=uxf0grb9&dl=1",
        "pedestrian_network": "https://www.dropbox.com/scl/fi/4jehpo3wqhzsh5piu4fju/cdmx_pedestrian_network.gpkg?rlkey=y5a5qcogv30b8ei3xjmg0tg7d&st=repl2kw2&dl=1",
        "population_count": "https://www.dropbox.com/scl/fi/9zqpenj5of6dh3roe415d/ZM_CDMX_pop_count.gpkg?rlkey=pveese4i72mzux0xnu83d3emu&st=sg1395re&dl=1",
        "poi_food": "https://www.dropbox.com/scl/fi/7jn6b25n7bw19a7t432da/ZM_CDMX_poi_food.gpkg?rlkey=zz6yn05b6cdhozltwk8y7h5un&st=hqkvqaqb&dl=1",
        "grid": "https://www.dropbox.com/scl/fi/1l8vls67z508zbtyc59lh/CDMX_ZM_grid.gpkg?rlkey=15eg31udo3vwbxll3gtxgpeqh&st=6sjf6i1m&dl=1",
        "centroids": "https://www.dropbox.com/scl/fi/fuodav4nm53vvtrwrfmsa/CDMX_ZM_centroids.gpkg?rlkey=wt40sbusj1vs88cz6k431gn2h&st=c3o6o2sn&dl=1"
    },

    "Monterrey": {
        "boundary": "https://www.dropbox.com/scl/fi/np2nkbtkgat78bbr0s7dw/Monterrey_ZM.gpkg?rlkey=0jk3znfz0rk9tg5m6a8ikmfn0&st=2azh8xih&dl=1",
        "pedestrian_network": "https://www.dropbox.com/scl/fi/21kaaeudrojptosjhl1w6/monterrey_pedestrian_network.gpkg?rlkey=b1du48lfj1kx77xyykrisdifu&st=mr8ybkt9&dl=1",
        "population_count": "https://www.dropbox.com/scl/fi/jn4b2k2g7cfdjv8axzgwh/ZM_monterrey_pop_count.gpkg?rlkey=azp2vyh96msarizx8740l5k3e&st=628u61gl&dl=1",
        "poi_food": "https://www.dropbox.com/scl/fi/51jyf0yqqw76te1halbki/ZM_monterrey_poi_food.gpkg?rlkey=xj4vn4mjixs05v30t0j9yk40o&st=kbbrvgd7&dl=1",
        "grid": "https://www.dropbox.com/scl/fi/9l3azcn83xl4shwx6kyk3/Monterrey_ZM_grid.gpkg?rlkey=xidswz5jiptkstxenrxy4bmtu&st=x5tpdfaj&dl=1",
        "centroids": "https://www.dropbox.com/scl/fi/da4wa3m2zi156gid0l1hr/Monterrey_ZM_centroids.gpkg?rlkey=caq151syyznb1e1w8mnfhb20g&st=cv9ym66w&dl=1"
    },

    # "Guadalajara": {
    #     "boundary": "https://www.dropbox.com/scl/fi/64c5zr15skwcjzeg6zize/Guadalajara_ZM.gpkg?rlkey=lbbof4a3zg2wk2qza62sepywj&st=zqnn7ter&dl=1",
    #     "pedestrian_network": "https://www.dropbox.com/scl/fi/nn6k3oqmtnlqhh495gf2g/guadalajara_pedestrian_network.gpkg?rlkey=evta4lq5sich8n4bxi32acq9m&st=7q6v5ch4&dl=1",
    #     "population_count": "https://www.dropbox.com/scl/fi/oeagwv2v3tb8bhnft5p58/ZM_guadalajara_pop_count.gpkg?rlkey=ksvf7q4f1kuzrrbma698jevaj&st=2wuyetz8&dl=1",
    #     "poi_food": "https://www.dropbox.com/scl/fi/1lek1qup0pwm3zr86hn45/ZM_guadalajara_poi_food.gpkg?rlkey=a3s3ly0bh3gld2twcezb9fz4r&st=zv8ibce5&dl=1",
    #     "grid": "https://www.dropbox.com/scl/fi/3o6sigukwuh7houty9b9t/Guadalajara_ZM_grid.gpkg?rlkey=w0b5jyk8296gaer2dbhug8yxf&st=1wpl6k4z&dl=1",
    #     "centroids": "https://www.dropbox.com/scl/fi/3o6sigukwuh7houty9b9t/Guadalajara_ZM_grid.gpkg?rlkey=w0b5jyk8296gaer2dbhug8yxf&st=7b9trc63&dl=1"
    # },

    # "Puebla-Tlaxcala": {
    #     "boundary": "https://www.dropbox.com/scl/fi/ugriorbjmkik5cbn2wv37/Puebla-Tlaxcala_ZM.gpkg?rlkey=p4hs2og0pkjkzch1phfhiytwr&st=x98uvwyr&dl=1",
    #     "pedestrian_network": "https://www.dropbox.com/scl/fi/j9sk44omxxivz1ny5809z/zacatecas_pedestrian_network.gpkg?rlkey=9n7mkh3lyfqs9f5jyak1ha69r&st=hvangc3i&dl=1",
    #     "population_count": "https://www.dropbox.com/scl/fi/5pi99ng04a80y73955l65/ZM_puebla-tlaxcala_pop_count.gpkg?rlkey=inlidxgrx006s33fl36722au8&st=wydtj805&dl=1",
    #     "poi_food": "https://www.dropbox.com/scl/fi/kiondyp29elyzuax5lh3k/ZM_puebla-tlaxcala_poi_food.gpkg?rlkey=5235edy8kuuadoz58ej7cgn9b&st=oink616f&dl=1",
    #     "grid": "https://www.dropbox.com/scl/fi/fm1slql9i3ykc9gqp4y65/Puebla-Tlaxcala_ZM_grid.gpkg?rlkey=m5b50dti53nfg9kmfuu48e7qx&st=sileju1l&dl=1",
    #     "centroids": "https://www.dropbox.com/scl/fi/t451nvx5eml3f1exve60k/Puebla-Tlaxcala_ZM_centroids.gpkg?rlkey=qx0a8ck5v3ynb4jr1o6vxh16q&st=5akxg2t2&dl=1"
    # },

    # "Toluca": {
    #     "boundary": "https://www.dropbox.com/scl/fi/x7ousiufbx5nwl5x74bto/Toluca_ZM.gpkg?rlkey=elw2ncwapp4auhzqxwdmwrj4e&st=eto6woz5&dl=1",
    #     "pedestrian_network": "https://www.dropbox.com/scl/fi/mrrfds6q9sbybd1f00act/toluca_pedestrian_network.gpkg?rlkey=2rmv86oqdobfmsvm142vuc3p5&st=0l1y7cj8&dl=1",
    #     "population_count": "https://www.dropbox.com/scl/fi/b00774yduf2v41aphycck/ZM_toluca_pop_count.gpkg?rlkey=fhptvs6xk27152qaqrighgrx3&st=8yisa2hs&dl=1",
    #     "poi_food": "https://www.dropbox.com/scl/fi/a99gcd0qirgm1hajpt6d9/ZM_toluca_poi_food.gpkg?rlkey=w1aljl0kt45k869kwub6d22bv&st=nydsc729&dl=1",
    #     "grid": "https://www.dropbox.com/scl/fi/vrrxnoiaqahrorrhd2rde/Toluca_ZM_grid.gpkg?rlkey=1ju8mzlrolzr3mtbfnryj0vc5&st=4j2a8as0&dl=1",
    #     "centroids": "https://www.dropbox.com/scl/fi/yurrz9670bd4th51dk7vp/Toluca_ZM_centroids.gpkg?rlkey=igwpr2ftctva28bo4vio9s4xk&st=1dkytwlq&dl=1"
    # },

    # "Merida": {
    #     "boundary": "https://www.dropbox.com/scl/fi/sfx5icr2o6hcvedc3s375/Merida_ZM.gpkg?rlkey=xft1cobuontr00ci2o7808cz8&st=ritq2j06&dl=1",
    #     "pedestrian_network": "https://www.dropbox.com/scl/fi/rxh87eyf32qodjcue9gij/merida_pedestrian_network.gpkg?rlkey=yjrjywxbew6b0nndsn1l0gx85&st=ymvu577h&dl=1",
    #     "population_count": "https://www.dropbox.com/scl/fi/ss2sb4ku8gq2n5le7v14t/ZM_merida_pop_count.gpkg?rlkey=nhs2fh2l39egvc9dr6xbyrsb4&st=xkwd14im&dl=1",
    #     "poi_food": "https://www.dropbox.com/scl/fi/b551b063nmfai0rjnzs6g/ZM_merida_poi_food.gpkg?rlkey=a4glsyy0edi3v8am1lzzib5li&st=b1cqjt2x&dl=1",
    #     "grid": "https://www.dropbox.com/scl/fi/6sarpx8rbnheh6waqqxpe/Merida_ZM_grid.gpkg?rlkey=306z45f8cmip99ny4zj5nb8pw&st=u7m42gek&dl=1",
    #     "centroids": "https://www.dropbox.com/scl/fi/mmcj20bd6nx1cwzbvpc8l/Merida_ZM_centroids.gpkg?rlkey=pgjx5k8ioz3jesmcds5vgs81e&st=d7o96o7f&dl=1"
    # },

    # "Tlaxcala-Apizaco": {
    #     "boundary": "https://www.dropbox.com/scl/fi/3zxzd839968tzrey5oe20/Tlaxcala-Apizaco_ZM.gpkg?rlkey=1lu0ok6w3q0b76rxyk6ugfiqf&st=pmp9vi7u&dl=1",
    #     "pedestrian_network": "https://www.dropbox.com/scl/fi/d01oru5la22m2gq5i5vua/Tlaxcala-Apizaco_pedestrian_network.gpkg?rlkey=dtgs800eagfgmdhnp9j1blfyl&st=y6yltf2i&dl=1",
    #     "population_count": "https://www.dropbox.com/scl/fi/jbovify8rc6pctdbueoen/ZM_Tlaxcala-Apizaco_pop_count.gpkg?rlkey=i14olk7rtfkx1egm0juxe2wx6&st=9m076a42&dl=1",
    #     "poi_food": "https://www.dropbox.com/scl/fi/7gtgc6f22x6uz7ckdahaw/ZM_Tlaxcala-Apizaco_poi_food.gpkg?rlkey=9ltk3uepf3g8pnzi6a28y4him&st=xlc8gtez&dl=1",
    #     "grid": "https://www.dropbox.com/scl/fi/q6x5t73xiv986gkupgytr/Tlaxcala-Apizaco_ZM_grid.gpkg?rlkey=ars2hy82q8fpz5ln0uy6k158n&st=v8jjeo47&dl=1",
    #     "centroids": "https://www.dropbox.com/scl/fi/fovc47lk6hh4ppajjl9ng/Tlaxcala-Apizaco_ZM_centroids.gpkg?rlkey=fccclewcpm4ei0z9jhs5f0gov&st=xxibtfzh&dl=1"
    # },

    # "Tijuana": {
    #     "boundary": "https://www.dropbox.com/scl/fi/uvip3ravz1ky2udyhhtk3/Tijuana_ZM.gpkg?rlkey=5ghpcfzjnh6h6mmsrcbv51327&st=8c97njjq&dl=1",
    #     "pedestrian_network": "https://www.dropbox.com/scl/fi/dc89efbp4iqohy989bto6/tijuana_pedestrian_network.gpkg?rlkey=eb7tpq2pr7opuuainzqby0huy&st=v4wg6iit&dl=1",
    #     "population_count": "https://www.dropbox.com/scl/fi/hziyx0gf2myncb4wilih3/ZM_tijuana_pop_count.gpkg?rlkey=6um1n6w6fbcekpu1k0vvu2nil&st=x9hy6cq9&dl=1",
    #     "poi_food": "https://www.dropbox.com/scl/fi/dlih2n7kqknuccgd0s5ub/ZM_tijuana_poi_food.gpkg?rlkey=lunh13tjtnf6l6x8oetcfr3tb&st=usamjfrf&dl=1",
    #     "grid": "https://www.dropbox.com/scl/fi/iemg8427nmml2hl2ji39i/Tijuana_ZM_grid.gpkg?rlkey=4jft3stjgedhe6yh8jg6oox9l&st=963o149j&dl=1",
    #     "centroids": "https://www.dropbox.com/scl/fi/wxb0vwfk4r43xlaqyr8jn/Tijuana_ZM_centroids.gpkg?rlkey=b0npl9dlabplfi7vtybjuxgsz&st=r8jz18gm&dl=1"
    # },

    # "Leon": {
    #     "boundary": "https://www.dropbox.com/scl/fi/23p3sebe9wwwycs3mo9ld/Leon_ZM.gpkg?rlkey=7v07bd6pqdhcztxjj6otlzwto&st=x3nqaima&dl=1",
    #     "pedestrian_network": "https://www.dropbox.com/scl/fi/cx3r1l4fzyxtuutba9epl/leon_pedestrian_network.gpkg?rlkey=jq4hiwqpjiiku5ebgn6j8tovq&st=9olygjkc&dl=1",
    #     "population_count": "https://www.dropbox.com/scl/fi/oyw4yhk4hx6bas3u3fxx7/ZM_leon_pop_count.gpkg?rlkey=di2u4wmvz6mlvdav69xnsbo2a&st=jsqx02kk&dl=1",
    #     "poi_food": "https://www.dropbox.com/scl/fi/sqdvgujveaoewyoku9sd0/ZM_leon_poi_food.gpkg?rlkey=wqhiql4y2z1o77t7wsmfzmmoz&st=7mbknyzv&dl=1",
    #     "grid": "https://www.dropbox.com/scl/fi/vyfk9spgc7q9mi343uoh4/Leon_ZM_grid.gpkg?rlkey=t5qbj5ovnzizweok7wp9k5lya&st=k082s705&dl=1",
    #     "centroids": "https://www.dropbox.com/scl/fi/a26cf9vyqq3riz4nurrxx/Leon_ZM_centroids.gpkg?rlkey=x9qrioj9eqsjkctx40c3i81ea&st=w79si7ia&dl=1"
    # },

    # "La Laguna": {
    #     "boundary": "https://www.dropbox.com/scl/fi/rtj11ma1g7ev0prpl0elf/Laguna_ZM.gpkg?rlkey=s2953b77p39sp04p2kl7wu936&st=foky2ig5&dl=1",
    #     "pedestrian_network": "https://www.dropbox.com/scl/fi/gcg6eu9i5pyyyowqu8m2g/laguna_pedestrian_network.gpkg?rlkey=qid3gws0ic2rdfyj3lz2eh7tl&st=09gtuge1&dl=1",
    #     "population_count": "https://www.dropbox.com/scl/fi/obauyohh8svjhu7z9olb3/ZM_laguna_pop_count.gpkg?rlkey=ndb7rcpd25mmuzurl0r5ws41g&st=847603tg&dl=1",
    #     "poi_food": "https://www.dropbox.com/scl/fi/ku96inlr4kvrtyqot9x98/ZM_laguna_poi_food.gpkg?rlkey=blfgcrfwq5selsg2l18w7ft1i&st=q7jhyink&dl=1",
    #     "grid": "https://www.dropbox.com/scl/fi/jr18ol0vv5k5ekymvivgd/Laguna_ZM_grid.gpkg?rlkey=vnqlmympbnu68sfuvtdinoxv4&st=qclq6g1c&dl=1",
    #     "centroids": "https://www.dropbox.com/scl/fi/u7u7z7x4fol9i0ndacqx4/Laguna_ZM_centroids.gpkg?rlkey=lyux2495vn5amxt1v7jlr6ozt&st=1zwyydi2&dl=1"
    # },

    # "Queretaro": {
    #     "boundary": "https://www.dropbox.com/scl/fi/h5oo7jrfkg146i9urm0i7/Queretaro_ZM.gpkg?rlkey=1hs3pn4me5mnqveg8xdlhfutx&st=qhccuqv2&dl=1",
    #     "pedestrian_network": "https://www.dropbox.com/scl/fi/fal6u5a1jb7igr0x012e2/queretaro_pedestrian_network.gpkg?rlkey=3zgofsbc0oj2a03l9nqc2iq96&st=h1bx74hs&dl=1",
    #     "population_count": "https://www.dropbox.com/scl/fi/t0qa7zijv60dad1rodqag/ZM_queretaro_pop_count.gpkg?rlkey=eheg5lt9j2aguuf4ofvzhj4l3&st=fsrvgpb7&dl=1",
    #     "poi_food": "https://www.dropbox.com/scl/fi/rbfyiq7f00dvpccca7r4m/ZM_queretaro_poi_food.gpkg?rlkey=4a1ne13mruvgb64p7onoele0j&st=yhqlku75&dl=1",
    #     "grid": "https://www.dropbox.com/scl/fi/fkhlsb226d1x9spirc47l/Queretaro_ZM_grid.gpkg?rlkey=eemz0bxofob6r62n23x411kvc&st=4we64nw7&dl=1",
    #     "centroids": "https://www.dropbox.com/scl/fi/w48l1qtoe3yyfzhsf7lq6/Queretaro_ZM_centroids.gpkg?rlkey=hx93d23vigyfnb0p4xy83y416&st=6mwlmm27&dl=1"
    # },

    # "Chihuahua": {
    #     "boundary": "https://www.dropbox.com/scl/fi/12tstdj7hwde7z79j2tqd/Chihuahua_ZM.gpkg?rlkey=iu7pqm2ezz1w9oqcf8zkrewe8&st=fq9nk545&dl=1",
    #     "pedestrian_network": "https://www.dropbox.com/scl/fi/fnsb4orldwj1z3vdbxnil/chihuahua_pedestrian_network.gpkg?rlkey=xhoilw6jkr3g9h9budgp3h5ki&st=ofv70l10&dl=1",
    #     "population_count": "https://www.dropbox.com/scl/fi/ir3l3wmc9lxfh9c75lfi1/ZM_chihuahua_pop_count.gpkg?rlkey=h7pjobhgb69s2oe9ihfsxohkj&st=kyan05c2&dl=1",
    #     "poi_food": "https://www.dropbox.com/scl/fi/enb3q6jv9bi8f5yp5htyd/ZM_chihuahua_poi_food.gpkg?rlkey=ke4dkbsq566kts5k6hcs1845q&st=xa13yjf0&dl=1",
    #     "grid": "https://www.dropbox.com/scl/fi/yzl70caf6j8tlc7cfvhrm/Chihuahua_ZM_grid.gpkg?rlkey=bpwgk3n391obudlygf7vbvncm&st=aq0ye21k&dl=1",
    #     "centroids": "https://www.dropbox.com/scl/fi/w8jofj0fvat26kv1r126s/Chihuahua_ZM_centroids.gpkg?rlkey=2lnr36p531vaiso6u5l0a24hf&st=9djqmmay&dl=1"
    # },

    # "Saltillo": {
    #     "boundary": "https://www.dropbox.com/scl/fi/50llro911fj53q1mqmgzh/Saltillo_ZM.gpkg?rlkey=evmgp9c3wot0d93ood7c3gi1w&st=2gj802h9&dl=1",
    #     "pedestrian_network": "https://www.dropbox.com/scl/fi/siqif4adlwlt8oe8zv1st/saltillo_pedestrian_network.gpkg?rlkey=b9rnc2fb2brtuf4i4qgntf316&st=6s1qw2dj&dl=1",
    #     "population_count": "https://www.dropbox.com/scl/fi/q7jbor8xdws7wr43o31d9/ZM_saltillo_pop_count.gpkg?rlkey=w42jpv1ui7cek0dnix9m9t2o9&st=jemkr6xs&dl=1",
    #     "poi_food": "https://www.dropbox.com/scl/fi/cbthzx3ugw06q7a88wwb4/ZM_saltillo_poi_food.gpkg?rlkey=znkpipezvuet7vfl38795rdkq&st=lq9n7ofy&dl=1",
    #     "grid": "https://www.dropbox.com/scl/fi/m5mjmsrzsh0h85fwu48h2/Saltillo_ZM_grid.gpkg?rlkey=tzkquyj02lz53qqbq6xybmwj8&st=xtht1qys&dl=1",
    #     "centroids": "https://www.dropbox.com/scl/fi/sp0p3dchkwewrlbh1nkk7/Saltillo_ZM_centroids.gpkg?rlkey=wqu0gpr8iz7w1vaa35yiygxgc&st=9kwk19gj&dl=1"
    # },

    # "Aguascalientes": {
    #     "boundary": "https://www.dropbox.com/scl/fi/kw674cefpxo8wij0zdqe9/Aguascalientes_ZM.gpkg?rlkey=j45a2sd79o55w7pnlgphbwy59&st=jd6dcv1o&dl=1",
    #     "pedestrian_network": "https://www.dropbox.com/scl/fi/9tq849alcusg2ag47sraa/aguascalientes_pedestrian_network.gpkg?rlkey=xsh2jawtod9ggekfhtmkpftt0&st=wcadiz91&dl=1",
    #     "population_count": "https://www.dropbox.com/scl/fi/hucfhjo92ei2vi0ja9zuo/ZM_aguascalientes_pop_count.gpkg?rlkey=8jgsfya7uqbevm3tfrue95dhe&st=dkvh1wx1&dl=1",
    #     "poi_food": "https://www.dropbox.com/scl/fi/yp439bd52sdzhweny7w9n/ZM_aguascalientes_poi_food.gpkg?rlkey=ga9x5skkdq48qqnxvmgklrm35&st=31yeltcr&dl=1",
    #     "grid": "https://www.dropbox.com/scl/fi/xuzfryq9mgncwbx43xpaw/Aguascalientes_ZM_grid.gpkg?rlkey=rwyzgaaezu7bicidnb0tra2hj&st=p3m4y2di&dl=1",
    #     "centroids": "https://www.dropbox.com/scl/fi/qdzmnup9hco7k3p4bkji3/Aguascalientes_ZM_centroids.gpkg?rlkey=aqidp949rus1k8yvawraaesui&st=2j8bxdc1&dl=1"
    # }
}
