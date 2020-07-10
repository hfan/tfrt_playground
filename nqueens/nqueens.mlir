func @nqueen() {
  %ch = hex.new.chain

  %n_queens          = hex.constant.i32 14
  %n_schedule_levels = hex.constant.i32 4


  %n_sols = "nqueens.i32"(%n_queens, %n_schedule_levels) : (i32, i32) -> i32

  hex.print.i32 %n_sols, %ch

  hex.return
}
