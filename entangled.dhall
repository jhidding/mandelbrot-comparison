let entangled = https://raw.githubusercontent.com/entangled/entangled/v1.2.2/data/config-schema.dhall
                sha256:9bb4c5649869175ad0b662d292fd81a3d5d9ccb503b1c7e316d531b7856fb096

let languages = entangled.languages #
    [ { name = "Bash", identifiers = ["bash", "sh"], comment = entangled.comments.hash }
    , { name = "INI", identifiers = ["ini", "config"], comment = entangled.comments.lispStyle }
    , { name = "sqlite3", identifiers = ["sqlite", "sqlite3"], comment = entangled.comments.haskellStyle }
    ]

in { entangled = entangled.Config ::
        { watchList = [ "docs/src/*.md" ] : List Text
        , languages = languages
        }
   }

