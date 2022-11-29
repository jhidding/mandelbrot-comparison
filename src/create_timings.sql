-- ~\~ language=sqlite3 filename=src/create_timings.sql
-- ~\~ begin <<docs/src/index.md|src/create_timings.sql>>[init]
drop table if exists "timings"
create table "timings" (
    "program" text not null,
    "language" text not null,
    "case" text not null,
    "parallel" integer not null,
    "runtime" real not null
);
-- ~\~ end
